#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
    ____  _____    _    ____  _____ ____  
   |  _ \| ____|  / \  |  _ \| ____|  _ \ 
   | |_) |  _|   / _ \ | |_) |  _| | |_) |
   |  _ <| |___ / ___ \|  __/| |___|  _ < 
   |_| \_\_____/_/   \_\_|   |_____|_| \_\
                                          
    REAPER :: Autonomous Kitchen Disruption Framework
    ================================================
    
    Carnegie Cookoff 2026 Entry
    Team HareshM - Kitchen Chaos Division
    
    Development Philosophy:
    -----------------------
    "Control the cooking infrastructure, control the game."
    
    This bot employs a dual-phase operational paradigm:
    Phase I  - Maximize order throughput via optimized task pipelining  
    Phase II - Strategic resource denial through targeted equipment seizure
    
    The key insight driving our sabotage timing is that opponents who never
    engage in counter-sabotage (i.e., purely cooperative bots) can be
    exploited by removing their cooking equipment at the critical moment.
    
    Version History:
    ----------------
    v0.1 - Initial prototype with basic order handling
    v0.2 - Added BFS pathfinding cache (huge perf gain!)  
    v0.3 - Implemented disruption protocol
    v0.4 - Fixed bug where we were checking our own map after switching lol
    v0.5 - Tuned timing parameters, added logging
    
    TODO(haresh): Consider adaptive switch timing based on score differential
    TODO(haresh): Maybe add plate priority after all pans stolen?
    
================================================================================
"""

# =============================================================================
#  DEPENDENCY IMPORTS - Grouped by origin
# =============================================================================

# Standard library modules
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict, 
    FrozenSet,
    List, 
    Optional, 
    Set, 
    Tuple,
    Union,
)

# Game engine modules  
from game_constants import (
    FoodType,
    GameConstants,
    ShopCosts,
    Team,
    TileType,
)
from item import Food, Pan, Plate
from robot_controller import RobotController


# =============================================================================
#  CONFIGURATION & TUNING CONSTANTS
# =============================================================================

# Diagnostic output toggle - disable for tournament submission
DIAGNOSTIC_OUTPUT_ENABLED: bool = True

# Prefix tag for log messages (helps grep through mixed logs)
_LOG_PREFIX: str = "[REAPER]"

# Ingredient metadata registry
# Maps ingredient names to their processing requirements and shop costs
INGREDIENT_METADATA_TABLE: Dict[str, Dict[str, Union[int, bool]]] = {
    'SAUCE':   {'purchaseCost': 2,  'requiresChopping': False, 'requiresCooking': False},
    'EGG':     {'purchaseCost': 10, 'requiresChopping': False, 'requiresCooking': True},
    'ONIONS':  {'purchaseCost': 4,  'requiresChopping': True,  'requiresCooking': False},
    'NOODLES': {'purchaseCost': 3,  'requiresChopping': False, 'requiresCooking': False},
    'MEAT':    {'purchaseCost': 12, 'requiresChopping': True,  'requiresCooking': True},
}

# Timing thresholds for order feasibility assessment
MINIMUM_TURNS_FOR_ORDER_VIABILITY: int = 20

# Sabotage operational window duration
DISRUPTION_PHASE_DURATION_TURNS: int = 60

# How early in switch window to initiate (first N turns)
DISRUPTION_INITIATION_WINDOW: int = 5


# =============================================================================
#  OPERATIONAL STATE IDENTIFIERS
# =============================================================================

# Using string constants for operational phases - easier to debug than ints
# (Initially tried IntEnum but string repr in logs was more readable)

class OperationalMode:
    """Top-level operational mode indicators."""
    PRODUCTION_MODE = "MODE_PRODUCTION"
    DISRUPTION_MODE = "MODE_DISRUPTION"


# Task pipeline state machine tokens
TASK_STATE_AWAITING_ASSIGNMENT = "AWAITING_ASSIGNMENT"
TASK_STATE_ACQUIRING_COOKWARE = "ACQUIRING_COOKWARE"
TASK_STATE_DEPLOYING_COOKWARE = "DEPLOYING_COOKWARE"
TASK_STATE_ACQUIRING_DISHWARE = "ACQUIRING_DISHWARE"  
TASK_STATE_DEPLOYING_DISHWARE = "DEPLOYING_DISHWARE"
TASK_STATE_ACQUIRING_INGREDIENT = "ACQUIRING_INGREDIENT"
TASK_STATE_STAGING_FOR_PREPARATION = "STAGING_FOR_PREP"
TASK_STATE_EXECUTING_PREPARATION = "EXECUTING_PREP"
TASK_STATE_RETRIEVING_PREPARED = "RETRIEVING_PREPARED"
TASK_STATE_INITIATING_HEAT_TREATMENT = "INITIATING_HEAT"
TASK_STATE_MONITORING_HEAT_TREATMENT = "MONITORING_HEAT"
TASK_STATE_EXTRACTING_FROM_COOKWARE = "EXTRACTING_COOKED"
TASK_STATE_DISPOSING_BURNT_ITEM = "DISPOSING_BURNT"
TASK_STATE_ASSEMBLING_ON_DISHWARE = "ASSEMBLING_DISH"
TASK_STATE_RETRIEVING_COMPLETED_DISH = "RETRIEVING_DISH"
TASK_STATE_DELIVERING_TO_CUSTOMER = "DELIVERING"
TASK_STATE_DISRUPTION_PATROL = "DISRUPTION_ACTIVE"


# =============================================================================
#  HELPER DATA STRUCTURES
# =============================================================================

@dataclass
class GridCoordinate:
    """
    Immutable 2D coordinate on the kitchen grid.
    
    Attributes
    ----------
    col : int
        Horizontal position (x-axis)
    row : int  
        Vertical position (y-axis)
    """
    col: int
    row: int
    
    def asTuple(self) -> Tuple[int, int]:
        return (self.col, self.row)
    
    def chebyshevDistanceTo(self, other: 'GridCoordinate') -> int:
        """Compute Chebyshev (chessboard) distance to another coordinate."""
        return max(abs(self.col - other.col), abs(self.row - other.row))


@dataclass  
class OrderExecutionContext:
    """
    Tracks state for an in-progress order fulfillment.
    
    Attributes
    ----------
    orderReference : Dict
        The order dict from game engine
    orderIdentifier : int
        Unique order ID for dedup
    completedIngredients : Set[str]
        Ingredient names already added to plate
    dishwareDeployed : bool
        Whether plate has been placed on assembly counter
    assemblyLocation : Optional[Tuple[int, int]]
        Counter position where dish is being assembled
    """
    orderReference: Dict = field(default_factory=dict)
    orderIdentifier: int = -1
    completedIngredients: Set[str] = field(default_factory=set)
    dishwareDeployed: bool = False
    assemblyLocation: Optional[Tuple[int, int]] = None


# =============================================================================
#  DIAGNOSTIC LOGGING FACILITY  
# =============================================================================

def _emitDiagnostic(messageContent: str) -> None:
    """
    Emit diagnostic message to stdout if enabled.
    
    Parameters
    ----------
    messageContent : str
        The message to log
        
    Notes
    -----
    Prefixes all messages with _LOG_PREFIX for easy filtering.
    In tournament mode, set DIAGNOSTIC_OUTPUT_ENABLED = False.
    """
    if DIAGNOSTIC_OUTPUT_ENABLED:
        print(f"{_LOG_PREFIX} {messageContent}")


# =============================================================================
#  PRIMARY BOT CONTROLLER CLASS
# =============================================================================

class BotPlayer:
    """
    Main bot controller implementing the REAPER strategy.
    
    This class manages all bot decision-making including:
    - Navigation via precomputed BFS distance matrices
    - Order selection and task pipeline execution
    - Disruption protocol during switch windows
    
    Parameters
    ----------
    mapReference : Map
        Reference to the game map object
        
    Attributes
    ----------
    kitchenGrid : Map
        Cached map reference
    gridWidthCells : int
        Map width in tiles
    gridHeightCells : int  
        Map height in tiles
    """
    
    def __init__(self, mapReference):
        # Core map geometry caching
        self.kitchenGrid = mapReference
        self.gridWidthCells = mapReference.width
        self.gridHeightCells = mapReference.height
        self.assignedTeam: Optional[Team] = None
        
        # Build tile position index
        # Maps tile type names to list of coordinates
        self.tileLocationRegistry: Dict[str, List[Tuple[int, int]]] = {}
        self.traversableCells: Set[Tuple[int, int]] = set()
        
        for xCoord in range(self.gridWidthCells):
            for yCoord in range(self.gridHeightCells):
                currentTile = mapReference.tiles[xCoord][yCoord]
                tileName = currentTile.tile_name
                
                if tileName not in self.tileLocationRegistry:
                    self.tileLocationRegistry[tileName] = []
                self.tileLocationRegistry[tileName].append((xCoord, yCoord))
                
                if currentTile.is_walkable:
                    self.traversableCells.add((xCoord, yCoord))
        
        # Initialize pathfinding infrastructure
        self._adjacentCellCache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self._distanceMatrices: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self._optimalFirstSteps: Dict[Tuple[int, int], Dict[Tuple[int, int], Tuple[int, int]]] = {}
        self._initializePathfindingMatrices()
        
        # Cache frequently-accessed tile locations
        self._shopTileLocations = self.tileLocationRegistry.get('SHOP', [])
        self._submissionTileLocations = self.tileLocationRegistry.get('SUBMIT', [])
        self._counterTileLocations = self.tileLocationRegistry.get('COUNTER', [])
        self._cookerTileLocations = self.tileLocationRegistry.get('COOKER', [])
        self._disposalTileLocations = self.tileLocationRegistry.get('TRASH', [])
        self._sinkTableTileLocations = self.tileLocationRegistry.get('SINK_TABLE', [])
        
        # Operational state tracking
        self.currentOperationalMode = OperationalMode.PRODUCTION_MODE
        self.hasInitiatedDisruption = False
        self.disruptionCommencementTurn = 0
        self.cookwareItemsSeized = 0
        self.dishwareItemsSeized = 0
        
        # Order management state
        self.activeOrderAssignments: Dict[int, OrderExecutionContext] = {}
        self.fulfilledOrderIdentifiers: Set[int] = set()
        self.botTaskPipelineState: Dict[int, Dict[str, Any]] = {}
        
        # Resource availability tracking
        self.cookwareHasBeenDeployed = False
        self.botAssemblyLocations: Dict[int, Tuple[int, int]] = {}
    
    # =========================================================================
    #  PATHFINDING SUBSYSTEM
    # =========================================================================
    
    def _initializePathfindingMatrices(self) -> None:
        """
        Precompute BFS distance matrices from every walkable cell.
        
        This runs once at initialization and builds O(N^2) lookup tables
        for instant distance queries during gameplay.
        
        Notes
        -----
        Also computes optimal first step towards each destination,
        enabling single-step navigation decisions in O(1) time.
        """
        for sourceCell in self.traversableCells:
            distanceFromSource: Dict[Tuple[int, int], int] = {sourceCell: 0}
            firstStepFromSource: Dict[Tuple[int, int], Tuple[int, int]] = {sourceCell: (0, 0)}
            
            explorationQueue = deque([sourceCell])
            
            while len(explorationQueue) > 0:
                currentX, currentY = explorationQueue.popleft()
                
                # Explore all 8 directions (including diagonals)
                for deltaX in (-1, 0, 1):
                    for deltaY in (-1, 0, 1):
                        # Skip the no-movement case
                        if deltaX == 0 and deltaY == 0:
                            continue
                            
                        neighborX = currentX + deltaX
                        neighborY = currentY + deltaY
                        neighborPos = (neighborX, neighborY)
                        
                        # Skip if not walkable or already visited
                        if neighborPos not in self.traversableCells:
                            continue
                        if neighborPos in distanceFromSource:
                            continue
                        
                        # Record distance and propagate first step
                        distanceFromSource[neighborPos] = distanceFromSource[(currentX, currentY)] + 1
                        
                        if (currentX, currentY) == sourceCell:
                            # This is the first step from source
                            firstStepFromSource[neighborPos] = (deltaX, deltaY)
                        else:
                            # Inherit first step from parent
                            firstStepFromSource[neighborPos] = firstStepFromSource[(currentX, currentY)]
                        
                        explorationQueue.append(neighborPos)
            
            self._distanceMatrices[sourceCell] = distanceFromSource
            self._optimalFirstSteps[sourceCell] = firstStepFromSource
    
    def _getAdjacentWalkableCells(
        self, 
        targetX: int, 
        targetY: int
    ) -> List[Tuple[int, int]]:
        """
        Find all walkable cells adjacent to a target tile.
        
        Parameters
        ----------
        targetX : int
            X coordinate of target tile
        targetY : int
            Y coordinate of target tile
            
        Returns
        -------
        List[Tuple[int, int]]
            List of adjacent walkable positions
            
        Notes
        -----
        Results are cached for efficiency.
        """
        cacheKey = (targetX, targetY)
        
        if cacheKey not in self._adjacentCellCache:
            adjacentCells = []
            
            for offsetX in range(-1, 2):
                for offsetY in range(-1, 2):
                    candidateX = targetX + offsetX
                    candidateY = targetY + offsetY
                    
                    if (candidateX, candidateY) in self.traversableCells:
                        adjacentCells.append((candidateX, candidateY))
            
            self._adjacentCellCache[cacheKey] = adjacentCells
        
        return self._adjacentCellCache[cacheKey]
    
    def _computeDistanceToTile(
        self,
        sourceX: int,
        sourceY: int, 
        targetX: int,
        targetY: int
    ) -> int:
        """
        Compute minimum distance from source to any cell adjacent to target.
        
        Parameters
        ----------
        sourceX, sourceY : int
            Current position
        targetX, targetY : int
            Target tile position (may be non-walkable)
            
        Returns
        -------
        int
            Minimum steps to reach adjacency, or 9999 if unreachable
        """
        adjacentCells = self._getAdjacentWalkableCells(targetX, targetY)
        
        if len(adjacentCells) == 0:
            return 9999
        
        sourcePos = (sourceX, sourceY)
        if sourcePos not in self._distanceMatrices:
            return 9999
        
        distanceTable = self._distanceMatrices[sourcePos]
        
        minimumDistance = 9999
        for adjacentPos in adjacentCells:
            distToAdjacent = distanceTable.get(adjacentPos, 9999)
            if distToAdjacent < minimumDistance:
                minimumDistance = distToAdjacent
        
        return minimumDistance
    
    def _locateNearestTileOfType(
        self,
        botX: int,
        botY: int,
        tileTypeName: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest tile of a given type to the bot.
        
        Parameters
        ----------
        botX, botY : int
            Current bot position
        tileTypeName : str
            The tile type to search for
            
        Returns
        -------
        Optional[Tuple[int, int]]
            Position of nearest matching tile, or None if none exist
        """
        candidateLocations = self.tileLocationRegistry.get(tileTypeName, [])
        
        if len(candidateLocations) == 0:
            return None
        
        nearestLocation = None
        nearestDistance = 999999
        
        for (locX, locY) in candidateLocations:
            dist = self._computeDistanceToTile(botX, botY, locX, locY)
            if dist < nearestDistance:
                nearestDistance = dist
                nearestLocation = (locX, locY)
        
        return nearestLocation
    
    def _navigateToTargetPosition(
        self,
        controller: RobotController,
        botIdentifier: int,
        targetPosition: Tuple[int, int]
    ) -> bool:
        """
        Move bot one step towards target. Returns True if already adjacent.
        
        Parameters
        ----------
        controller : RobotController
            Game controller reference
        botIdentifier : int
            ID of bot to move
        targetPosition : Tuple[int, int]
            Destination tile coordinates
            
        Returns
        -------
        bool
            True if bot is now adjacent to target (can interact)
        """
        botState = controller.get_bot_state(botIdentifier)
        if botState is None:
            return False
        
        currentX, currentY = botState['x'], botState['y']
        targetX, targetY = targetPosition
        
        # Check if already adjacent (Chebyshev distance <= 1)
        if max(abs(currentX - targetX), abs(currentY - targetY)) <= 1:
            return True
        
        # Find walkable cells adjacent to target
        adjacentCells = self._getAdjacentWalkableCells(targetX, targetY)
        if len(adjacentCells) == 0:
            return False
        
        currentPos = (currentX, currentY)
        if currentPos not in self._distanceMatrices:
            return False
        
        # Find closest adjacent cell
        distTable = self._distanceMatrices[currentPos]
        
        optimalDestination = None
        optimalDist = 999999
        for adjCell in adjacentCells:
            d = distTable.get(adjCell, 999999)
            if d < optimalDist:
                optimalDist = d
                optimalDestination = adjCell
        
        if optimalDestination is None or optimalDestination not in distTable:
            return False
        
        # Use precomputed first step if available
        firstStepTable = self._optimalFirstSteps.get(currentPos, {})
        if optimalDestination in firstStepTable:
            deltaX, deltaY = firstStepTable[optimalDestination]
            if controller.can_move(botIdentifier, deltaX, deltaY):
                controller.move(botIdentifier, deltaX, deltaY)
                newX, newY = currentX + deltaX, currentY + deltaY
                return max(abs(newX - targetX), abs(newY - targetY)) <= 1
        
        # Fallback: try any valid move
        for dX in (-1, 0, 1):
            for dY in (-1, 0, 1):
                if dX == 0 and dY == 0:
                    continue
                if controller.can_move(botIdentifier, dX, dY):
                    controller.move(botIdentifier, dX, dY)
                    return False
        
        return False
    
    # =========================================================================
    #  DISRUPTION PROTOCOL - Phase II Operations
    # =========================================================================
    
    def _executeDisruptionProtocol(
        self,
        controller: RobotController,
        botIdentifier: int,
        teamRef: Team
    ) -> None:
        """
        Execute aggressive resource denial operations.
        
        After switching to enemy map, systematically:
        1. Dispose of any held items to free hands
        2. Seize cookware from cooker stations (highest impact)
        3. Seize dishware from sink tables  
        4. Seize any items from counters
        5. Patrol near cookers waiting for new equipment
        
        Parameters
        ----------
        controller : RobotController
            Game controller
        botIdentifier : int
            Bot to control
        teamRef : Team
            Our team (for reference - we're on enemy map now)
        """
        botState = controller.get_bot_state(botIdentifier)
        if botState is None:
            return
        
        positionX = botState['x']
        positionY = botState['y']
        currentlyHolding = botState.get('holding')
        
        # CRITICAL: After switch, we're operating on enemy map!
        # Need to query tiles using enemy team reference
        mapTeamIdentifier = botState.get('map_team', teamRef.name)
        enemyTeamRef = controller.get_enemy_team()
        
        disposalLocation = self._disposalTileLocations[0] if self._disposalTileLocations else None
        
        # If holding anything, dispose immediately to free hands for seizure
        if currentlyHolding is not None:
            _emitDiagnostic(
                f"Bot #{botIdentifier} at ({positionX},{positionY}) holding "
                f"{currentlyHolding.get('type')}, routing to disposal at {disposalLocation}"
            )
            
            if disposalLocation is not None:
                distToDisposal = max(
                    abs(positionX - disposalLocation[0]), 
                    abs(positionY - disposalLocation[1])
                )
                _emitDiagnostic(f"  Distance to disposal unit: {distToDisposal}")
                
                if distToDisposal <= 1:
                    result = controller.trash(botIdentifier, disposalLocation[0], disposalLocation[1])
                    _emitDiagnostic(f"  Disposal result: {result}")
                else:
                    self._navigateToTargetPosition(controller, botIdentifier, disposalLocation)
                    _emitDiagnostic("  Navigating to disposal unit...")
            else:
                _emitDiagnostic("WARNING: No disposal unit found on map!")
            return
        
        # Priority 1: Seize cookware from cooker stations (MAXIMUM IMPACT)
        _emitDiagnostic(
            f"Bot #{botIdentifier} scanning {len(self._cookerTileLocations)} cooker stations "
            f"(enemy={enemyTeamRef.name}, mapTeam={mapTeamIdentifier})"
        )
        
        for cookerPos in self._cookerTileLocations:
            tileData = controller.get_tile(enemyTeamRef, cookerPos[0], cookerPos[1])
            _emitDiagnostic(
                f"  Cooker {cookerPos}: tile={tileData}, "
                f"item={getattr(tileData, 'item', None) if tileData else None}"
            )
            
            if tileData is not None:
                tileItem = getattr(tileData, 'item', None)
                if isinstance(tileItem, Pan):
                    _emitDiagnostic(f"COOKWARE LOCATED at {cookerPos}, initiating seizure...")
                    
                    if self._navigateToTargetPosition(controller, botIdentifier, cookerPos):
                        # If pan has cooked food, extract it first
                        if tileItem.food is not None and tileItem.food.cooked_stage >= 1:
                            if controller.take_from_pan(botIdentifier, cookerPos[0], cookerPos[1]):
                                _emitDiagnostic(f"Extracted contents from cookware at {cookerPos}")
                        else:
                            if controller.pickup(botIdentifier, cookerPos[0], cookerPos[1]):
                                self.cookwareItemsSeized += 1
                                _emitDiagnostic(f"COOKWARE SEIZED from {cookerPos}!")
                    return
        
        # Priority 2: Seize dishware from sink tables
        for sinkTablePos in self._sinkTableTileLocations:
            tileData = controller.get_tile(enemyTeamRef, sinkTablePos[0], sinkTablePos[1])
            
            if tileData is not None:
                tileItem = getattr(tileData, 'item', None)
                if isinstance(tileItem, Plate):
                    _emitDiagnostic(f"DISHWARE LOCATED at {sinkTablePos}, initiating seizure...")
                    
                    if self._navigateToTargetPosition(controller, botIdentifier, sinkTablePos):
                        if controller.pickup(botIdentifier, sinkTablePos[0], sinkTablePos[1]):
                            self.dishwareItemsSeized += 1
                            _emitDiagnostic(f"DISHWARE SEIZED from {sinkTablePos}!")
                    return
        
        # Priority 3: Seize anything from counters
        for counterPos in self._counterTileLocations:
            tileData = controller.get_tile(enemyTeamRef, counterPos[0], counterPos[1])
            
            if tileData is not None and getattr(tileData, 'item', None) is not None:
                _emitDiagnostic(f"ITEM LOCATED on counter {counterPos}, initiating seizure...")
                
                if self._navigateToTargetPosition(controller, botIdentifier, counterPos):
                    if controller.pickup(botIdentifier, counterPos[0], counterPos[1]):
                        _emitDiagnostic(f"ITEM SEIZED from counter {counterPos}!")
                return
        
        # Nothing to seize - patrol near cookers awaiting new equipment
        _emitDiagnostic(f"No seizure targets found, initiating patrol...")
        
        if len(self._cookerTileLocations) > 0:
            nearestCooker = None
            nearestDist = 999999
            
            for cookerPos in self._cookerTileLocations:
                d = self._computeDistanceToTile(positionX, positionY, cookerPos[0], cookerPos[1])
                if d < nearestDist:
                    nearestDist = d
                    nearestCooker = cookerPos
            
            if nearestCooker is not None:
                self._navigateToTargetPosition(controller, botIdentifier, nearestCooker)
    
    # =========================================================================
    #  ORDER SELECTION ENGINE
    # =========================================================================
    
    def _selectOptimalOrderForBot(
        self,
        controller: RobotController,
        teamRef: Team,
        botIdentifier: int
    ) -> Optional[Dict]:
        """
        Select the best available order for a bot to fulfill.
        
        Scoring prioritizes:
        - Simple orders (no cooking requirement)
        - High profit margin relative to time remaining
        - Single or dual ingredient orders
        
        Parameters
        ----------
        controller : RobotController
            Game controller
        teamRef : Team  
            Our team
        botIdentifier : int
            Bot making the selection
            
        Returns
        -------
        Optional[Dict]
            Best order to work on, or None if nothing suitable
        """
        availableOrders = controller.get_orders(teamRef)
        currentTurnNumber = controller.get_turn()
        currentFunds = controller.get_team_money(teamRef)
        
        optimalOrder = None
        optimalScore = -999.0
        
        for orderData in availableOrders:
            # Skip inactive or already completed orders
            if not orderData['is_active']:
                continue
            if orderData.get('completed_turn') is not None:
                continue
            if orderData['order_id'] in self.fulfilledOrderIdentifiers:
                continue
            
            # Skip orders already claimed by another bot
            isClaimedByOther = False
            for otherBotId, context in self.activeOrderAssignments.items():
                if otherBotId != botIdentifier:
                    if context.orderIdentifier == orderData['order_id']:
                        isClaimedByOther = True
                        break
            
            if isClaimedByOther:
                continue
            
            # Calculate total cost and processing requirements
            estimatedCost = ShopCosts.PLATE.buy_cost
            needsCookingStep = False
            needsChoppingStep = False
            
            for ingredientName in orderData['required']:
                ingredientMeta = INGREDIENT_METADATA_TABLE.get(ingredientName, {})
                estimatedCost += ingredientMeta.get('purchaseCost', 5)
                
                if ingredientMeta.get('requiresCooking', False):
                    needsCookingStep = True
                if ingredientMeta.get('requiresChopping', False):
                    needsChoppingStep = True
            
            # Add cookware cost if needed and not yet deployed
            if needsCookingStep and not self.cookwareHasBeenDeployed:
                estimatedCost += ShopCosts.PAN.buy_cost
            
            # Skip if insufficient funds
            if estimatedCost > currentFunds:
                continue
            
            # Check time feasibility
            turnsRemaining = orderData['expires_turn'] - currentTurnNumber
            if turnsRemaining < MINIMUM_TURNS_FOR_ORDER_VIABILITY:
                continue
            
            # Compute score - prioritize profitability per unit time
            projectedProfit = orderData['reward'] - estimatedCost
            orderScore = (projectedProfit / max(turnsRemaining, 1)) * 10.0
            
            # Apply complexity bonuses/penalties
            if not needsCookingStep:
                orderScore += 15.0  # Strong preference for non-cooking orders
            if not needsChoppingStep:
                orderScore += 5.0
            
            # Bonus for simpler orders
            ingredientCount = len(orderData['required'])
            if ingredientCount == 1:
                orderScore += 10.0
            elif ingredientCount == 2:
                orderScore += 5.0
            
            if orderScore > optimalScore:
                optimalScore = orderScore
                optimalOrder = orderData
        
        return optimalOrder
    
    # =========================================================================
    #  TILE QUERY UTILITIES
    # =========================================================================
    
    def _isTileCurrentlyEmpty(
        self,
        controller: RobotController,
        teamRef: Team,
        xPos: int,
        yPos: int
    ) -> bool:
        """Check if a tile has no item on it."""
        tileData = controller.get_tile(teamRef, xPos, yPos)
        if tileData is not None:
            return getattr(tileData, 'item', None) is None
        return True
    
    def _locateAvailableCounterSurface(
        self,
        controller: RobotController,
        teamRef: Team,
        botX: int,
        botY: int
    ) -> Optional[Tuple[int, int]]:
        """
        Find nearest empty counter for item placement.
        
        Parameters
        ----------
        controller : RobotController
            Game controller
        teamRef : Team
            Team to query
        botX, botY : int
            Bot's current position for distance calculation
            
        Returns
        -------
        Optional[Tuple[int, int]]
            Position of nearest empty counter, or None
        """
        # Sort counters by distance to bot
        sortedCounters = sorted(
            self._counterTileLocations,
            key=lambda pos: self._computeDistanceToTile(botX, botY, pos[0], pos[1])
        )
        
        for counterPos in sortedCounters:
            if self._isTileCurrentlyEmpty(controller, teamRef, counterPos[0], counterPos[1]):
                return counterPos
        
        return None
    
    # =========================================================================
    #  PRODUCTION PIPELINE - Phase I Operations  
    # =========================================================================
    
    def _executeProductionPipeline(
        self,
        controller: RobotController,
        botIdentifier: int,
        teamRef: Team
    ) -> None:
        """
        Execute order fulfillment task pipeline.
        
        Implements a multi-phase state machine:
        - IDLE -> Select order
        - BUY_PAN -> PLACE_PAN (if cooking needed)
        - BUY_PLATE -> PLACE_PLATE
        - BUY_INGREDIENT -> CHOP/COOK as needed -> ADD_TO_PLATE
        - PICKUP_PLATE -> SUBMIT
        
        Parameters
        ----------
        controller : RobotController
            Game controller  
        botIdentifier : int
            Bot executing the pipeline
        teamRef : Team
            Our team
        """
        botState = controller.get_bot_state(botIdentifier)
        if botState is None:
            return
        
        posX, posY = botState['x'], botState['y']
        currentlyHolding = botState.get('holding')
        availableFunds = controller.get_team_money(teamRef)
        
        # Get or initialize task state
        taskState = self.botTaskPipelineState.get(botIdentifier, {
            'phase': TASK_STATE_AWAITING_ASSIGNMENT
        })
        currentPhase = taskState.get('phase', TASK_STATE_AWAITING_ASSIGNMENT)
        
        # Cache key tile positions
        shopPos = self._shopTileLocations[0] if self._shopTileLocations else None
        submitPos = self._submissionTileLocations[0] if self._submissionTileLocations else None
        disposalPos = self._disposalTileLocations[0] if self._disposalTileLocations else None
        
        if shopPos is None or submitPos is None:
            return
        
        shopX, shopY = shopPos
        
        # Get current order context
        orderContext = self.activeOrderAssignments.get(botIdentifier)
        orderData = orderContext.orderReference if orderContext else None
        
        # =====================================================================
        # State: AWAITING_ASSIGNMENT - Select new order
        # =====================================================================
        if currentPhase == TASK_STATE_AWAITING_ASSIGNMENT:
            if orderData is None:
                selectedOrder = self._selectOptimalOrderForBot(controller, teamRef, botIdentifier)
                
                if selectedOrder is not None:
                    newContext = OrderExecutionContext(
                        orderReference=selectedOrder,
                        orderIdentifier=selectedOrder['order_id'],
                        completedIngredients=set(),
                        dishwareDeployed=False,
                        assemblyLocation=None
                    )
                    self.activeOrderAssignments[botIdentifier] = newContext
                    orderContext = newContext
                    orderData = selectedOrder
                    
                    _emitDiagnostic(
                        f"Bot #{botIdentifier} claimed order {selectedOrder['order_id']}: "
                        f"{selectedOrder['required']}"
                    )
            
            if orderData is not None:
                # Determine next phase based on order requirements
                requiresCookingEquipment = any(
                    INGREDIENT_METADATA_TABLE.get(ing, {}).get('requiresCooking', False)
                    for ing in orderData['required']
                )
                
                if requiresCookingEquipment and not self.cookwareHasBeenDeployed:
                    taskState['phase'] = TASK_STATE_ACQUIRING_COOKWARE
                elif not orderContext.dishwareDeployed:
                    taskState['phase'] = TASK_STATE_ACQUIRING_DISHWARE
                else:
                    # Find next ingredient to process
                    for ingredientName in orderData['required']:
                        if ingredientName not in orderContext.completedIngredients:
                            taskState['phase'] = TASK_STATE_ACQUIRING_INGREDIENT
                            taskState['targetIngredient'] = ingredientName
                            break
                    else:
                        taskState['phase'] = TASK_STATE_RETRIEVING_COMPLETED_DISH
        
        # =====================================================================
        # State: ACQUIRING_COOKWARE - Buy pan from shop
        # =====================================================================
        elif currentPhase == TASK_STATE_ACQUIRING_COOKWARE:
            if currentlyHolding is not None and currentlyHolding.get('type') == 'Pan':
                taskState['phase'] = TASK_STATE_DEPLOYING_COOKWARE
            elif currentlyHolding is None:
                if self._navigateToTargetPosition(controller, botIdentifier, shopPos):
                    if availableFunds >= ShopCosts.PAN.buy_cost:
                        controller.buy(botIdentifier, ShopCosts.PAN, shopX, shopY)
            else:
                # Holding wrong item, dispose it
                if disposalPos and self._navigateToTargetPosition(controller, botIdentifier, disposalPos):
                    controller.trash(botIdentifier, disposalPos[0], disposalPos[1])
        
        # =====================================================================
        # State: DEPLOYING_COOKWARE - Place pan on cooker
        # =====================================================================
        elif currentPhase == TASK_STATE_DEPLOYING_COOKWARE:
            if len(self._cookerTileLocations) > 0:
                cookerPos = self._cookerTileLocations[0]
                
                if self._navigateToTargetPosition(controller, botIdentifier, cookerPos):
                    if controller.place(botIdentifier, cookerPos[0], cookerPos[1]):
                        self.cookwareHasBeenDeployed = True
                        taskState['phase'] = TASK_STATE_AWAITING_ASSIGNMENT
        
        # =====================================================================
        # State: ACQUIRING_DISHWARE - Buy plate from shop
        # =====================================================================
        elif currentPhase == TASK_STATE_ACQUIRING_DISHWARE:
            if currentlyHolding is not None and currentlyHolding.get('type') == 'Plate':
                taskState['phase'] = TASK_STATE_DEPLOYING_DISHWARE
            elif currentlyHolding is None:
                if self._navigateToTargetPosition(controller, botIdentifier, shopPos):
                    if availableFunds >= ShopCosts.PLATE.buy_cost:
                        controller.buy(botIdentifier, ShopCosts.PLATE, shopX, shopY)
            else:
                if disposalPos and self._navigateToTargetPosition(controller, botIdentifier, disposalPos):
                    controller.trash(botIdentifier, disposalPos[0], disposalPos[1])
        
        # =====================================================================
        # State: DEPLOYING_DISHWARE - Place plate on counter
        # =====================================================================
        elif currentPhase == TASK_STATE_DEPLOYING_DISHWARE:
            assemblyLoc = orderContext.assemblyLocation if orderContext else None
            
            if assemblyLoc is None:
                assemblyLoc = self._locateAvailableCounterSurface(controller, teamRef, posX, posY)
                if orderContext is not None:
                    orderContext.assemblyLocation = assemblyLoc
            
            if assemblyLoc is not None:
                if self._navigateToTargetPosition(controller, botIdentifier, assemblyLoc):
                    if self._isTileCurrentlyEmpty(controller, teamRef, assemblyLoc[0], assemblyLoc[1]):
                        if controller.place(botIdentifier, assemblyLoc[0], assemblyLoc[1]):
                            if orderContext is not None:
                                orderContext.dishwareDeployed = True
                            taskState['phase'] = TASK_STATE_AWAITING_ASSIGNMENT
        
        # =====================================================================
        # State: ACQUIRING_INGREDIENT - Buy ingredient from shop
        # =====================================================================
        elif currentPhase == TASK_STATE_ACQUIRING_INGREDIENT:
            ingredientName = taskState.get('targetIngredient')
            foodTypeEnum = getattr(FoodType, ingredientName, None) if ingredientName else None
            ingredientMeta = INGREDIENT_METADATA_TABLE.get(ingredientName, {})
            
            if currentlyHolding is not None and currentlyHolding.get('type') == 'Food':
                if ingredientMeta.get('requiresChopping', False):
                    taskState['phase'] = TASK_STATE_STAGING_FOR_PREPARATION
                elif ingredientMeta.get('requiresCooking', False):
                    taskState['phase'] = TASK_STATE_INITIATING_HEAT_TREATMENT
                else:
                    taskState['phase'] = TASK_STATE_ASSEMBLING_ON_DISHWARE
            elif currentlyHolding is None:
                if self._navigateToTargetPosition(controller, botIdentifier, shopPos):
                    if foodTypeEnum is not None and availableFunds >= foodTypeEnum.buy_cost:
                        controller.buy(botIdentifier, foodTypeEnum, shopX, shopY)
            else:
                if disposalPos and self._navigateToTargetPosition(controller, botIdentifier, disposalPos):
                    controller.trash(botIdentifier, disposalPos[0], disposalPos[1])
        
        # =====================================================================
        # State: STAGING_FOR_PREPARATION - Place food on counter for chopping
        # =====================================================================
        elif currentPhase == TASK_STATE_STAGING_FOR_PREPARATION:
            counterPos = self._locateAvailableCounterSurface(controller, teamRef, posX, posY)
            
            if counterPos is not None:
                taskState['prepLocation'] = counterPos
                
                if self._navigateToTargetPosition(controller, botIdentifier, counterPos):
                    if controller.place(botIdentifier, counterPos[0], counterPos[1]):
                        taskState['phase'] = TASK_STATE_EXECUTING_PREPARATION
        
        # =====================================================================
        # State: EXECUTING_PREPARATION - Chop the food
        # =====================================================================
        elif currentPhase == TASK_STATE_EXECUTING_PREPARATION:
            prepLoc = taskState.get('prepLocation')
            
            if prepLoc is not None:
                if self._navigateToTargetPosition(controller, botIdentifier, prepLoc):
                    tileData = controller.get_tile(teamRef, prepLoc[0], prepLoc[1])
                    
                    if tileData is not None:
                        tileItem = getattr(tileData, 'item', None)
                        if isinstance(tileItem, Food):
                            if tileItem.chopped:
                                taskState['phase'] = TASK_STATE_RETRIEVING_PREPARED
                            else:
                                controller.chop(botIdentifier, prepLoc[0], prepLoc[1])
        
        # =====================================================================
        # State: RETRIEVING_PREPARED - Pick up chopped food
        # =====================================================================
        elif currentPhase == TASK_STATE_RETRIEVING_PREPARED:
            prepLoc = taskState.get('prepLocation')
            
            if prepLoc is not None:
                if self._navigateToTargetPosition(controller, botIdentifier, prepLoc):
                    if controller.pickup(botIdentifier, prepLoc[0], prepLoc[1]):
                        ingredientMeta = INGREDIENT_METADATA_TABLE.get(
                            taskState.get('targetIngredient'), {}
                        )
                        
                        if ingredientMeta.get('requiresCooking', False):
                            taskState['phase'] = TASK_STATE_INITIATING_HEAT_TREATMENT
                        else:
                            taskState['phase'] = TASK_STATE_ASSEMBLING_ON_DISHWARE
        
        # =====================================================================
        # State: INITIATING_HEAT_TREATMENT - Put food in pan
        # =====================================================================
        elif currentPhase == TASK_STATE_INITIATING_HEAT_TREATMENT:
            if len(self._cookerTileLocations) > 0:
                cookerPos = self._cookerTileLocations[0]
                
                if self._navigateToTargetPosition(controller, botIdentifier, cookerPos):
                    if controller.place(botIdentifier, cookerPos[0], cookerPos[1]):
                        taskState['phase'] = TASK_STATE_MONITORING_HEAT_TREATMENT
                        taskState['heatLocation'] = cookerPos
        
        # =====================================================================
        # State: MONITORING_HEAT_TREATMENT - Wait for cooking
        # =====================================================================
        elif currentPhase == TASK_STATE_MONITORING_HEAT_TREATMENT:
            heatLoc = taskState.get('heatLocation')
            
            if heatLoc is not None:
                tileData = controller.get_tile(teamRef, heatLoc[0], heatLoc[1])
                
                if tileData is not None:
                    tileItem = getattr(tileData, 'item', None)
                    if isinstance(tileItem, Pan):
                        if tileItem.food is not None:
                            cookingStage = getattr(tileItem.food, 'cooked_stage', 0)
                            if cookingStage >= 1:  # Cooked or burnt
                                taskState['phase'] = TASK_STATE_EXTRACTING_FROM_COOKWARE
        
        # =====================================================================
        # State: EXTRACTING_FROM_COOKWARE - Take food from pan
        # =====================================================================
        elif currentPhase == TASK_STATE_EXTRACTING_FROM_COOKWARE:
            heatLoc = taskState.get('heatLocation')
            
            if heatLoc is not None:
                if currentlyHolding is not None:
                    cookingStage = currentlyHolding.get('cooked_stage', 0)
                    if cookingStage == 2:  # Burnt!
                        taskState['phase'] = TASK_STATE_DISPOSING_BURNT_ITEM
                    else:
                        taskState['phase'] = TASK_STATE_ASSEMBLING_ON_DISHWARE
                elif self._navigateToTargetPosition(controller, botIdentifier, heatLoc):
                    controller.take_from_pan(botIdentifier, heatLoc[0], heatLoc[1])
        
        # =====================================================================
        # State: DISPOSING_BURNT_ITEM - Trash burnt food and retry
        # =====================================================================
        elif currentPhase == TASK_STATE_DISPOSING_BURNT_ITEM:
            if disposalPos is not None:
                if self._navigateToTargetPosition(controller, botIdentifier, disposalPos):
                    controller.trash(botIdentifier, disposalPos[0], disposalPos[1])
                    taskState['phase'] = TASK_STATE_ACQUIRING_INGREDIENT  # Retry
        
        # =====================================================================
        # State: ASSEMBLING_ON_DISHWARE - Add ingredient to plate
        # =====================================================================
        elif currentPhase == TASK_STATE_ASSEMBLING_ON_DISHWARE:
            assemblyLoc = orderContext.assemblyLocation if orderContext else None
            
            if assemblyLoc is not None:
                if self._navigateToTargetPosition(controller, botIdentifier, assemblyLoc):
                    if controller.place(botIdentifier, assemblyLoc[0], assemblyLoc[1]):
                        ingredientName = taskState.get('targetIngredient')
                        if ingredientName and orderContext:
                            orderContext.completedIngredients.add(ingredientName)
                        taskState['phase'] = TASK_STATE_AWAITING_ASSIGNMENT
        
        # =====================================================================
        # State: RETRIEVING_COMPLETED_DISH - Pick up finished plate
        # =====================================================================
        elif currentPhase == TASK_STATE_RETRIEVING_COMPLETED_DISH:
            assemblyLoc = orderContext.assemblyLocation if orderContext else None
            
            if assemblyLoc is not None:
                if self._navigateToTargetPosition(controller, botIdentifier, assemblyLoc):
                    if controller.pickup(botIdentifier, assemblyLoc[0], assemblyLoc[1]):
                        taskState['phase'] = TASK_STATE_DELIVERING_TO_CUSTOMER
        
        # =====================================================================
        # State: DELIVERING_TO_CUSTOMER - Submit completed order
        # =====================================================================
        elif currentPhase == TASK_STATE_DELIVERING_TO_CUSTOMER:
            submitX, submitY = submitPos
            
            if self._navigateToTargetPosition(controller, botIdentifier, submitPos):
                if controller.submit(botIdentifier, submitX, submitY):
                    if orderContext is not None:
                        self.fulfilledOrderIdentifiers.add(orderContext.orderIdentifier)
                        _emitDiagnostic(
                            f"Bot #{botIdentifier} COMPLETED order {orderContext.orderIdentifier}!"
                        )
                    
                    # Clear assignment and reset to idle
                    self.activeOrderAssignments.pop(botIdentifier, None)
                    taskState['phase'] = TASK_STATE_AWAITING_ASSIGNMENT
        
        # Save task state
        self.botTaskPipelineState[botIdentifier] = taskState
    
    # =========================================================================
    #  MAIN TURN CONTROLLER
    # =========================================================================
    
    def play_turn(self, controller: RobotController) -> None:
        """
        Main entry point called each game turn.
        
        Coordinates all bot activities based on current operational mode
        and handles phase transitions (Production -> Disruption).
        
        Parameters
        ----------
        controller : RobotController
            Game controller reference
        """
        currentTurnNumber = controller.get_turn()
        ourTeam = controller.get_team()
        
        # Cache team reference on first turn
        if self.assignedTeam is None:
            self.assignedTeam = ourTeam
        
        teamBotIds = controller.get_team_bot_ids(ourTeam)
        
        # Initialize task state for any new bots
        for botId in teamBotIds:
            if botId not in self.botTaskPipelineState:
                self.botTaskPipelineState[botId] = {
                    'phase': TASK_STATE_AWAITING_ASSIGNMENT
                }
        
        # Query switch window parameters
        switchWindowInfo = controller.get_switch_info()
        switchWindowStart = switchWindowInfo.get('switch_turn', 250)
        switchWindowDuration = switchWindowInfo.get('switch_duration', 100)
        
        currentlyInSwitchWindow = (
            switchWindowStart <= currentTurnNumber < switchWindowStart + switchWindowDuration
        )
        
        # =====================================================================
        # DISRUPTION INITIATION LOGIC
        # Switch to enemy map at start of window for maximum impact
        # =====================================================================
        if currentlyInSwitchWindow:
            if not self.hasInitiatedDisruption:
                if self.currentOperationalMode == OperationalMode.PRODUCTION_MODE:
                    if controller.can_switch_maps():
                        turnsIntoWindow = currentTurnNumber - switchWindowStart
                        
                        # Initiate within first few turns of window
                        if turnsIntoWindow < DISRUPTION_INITIATION_WINDOW:
                            if controller.switch_maps():
                                self.hasInitiatedDisruption = True
                                self.currentOperationalMode = OperationalMode.DISRUPTION_MODE
                                self.disruptionCommencementTurn = currentTurnNumber
                                
                                _emitDiagnostic(
                                    f"=== DISRUPTION PROTOCOL INITIATED at turn {currentTurnNumber} ==="
                                )
                                
                                # Reset all bots to disruption mode
                                for botId in teamBotIds:
                                    self.botTaskPipelineState[botId] = {
                                        'phase': TASK_STATE_DISRUPTION_PATROL
                                    }
        
        # =====================================================================
        # PHASE TRANSITION: Disruption -> Production
        # Return after spending designated time on enemy map
        # =====================================================================
        if self.currentOperationalMode == OperationalMode.DISRUPTION_MODE:
            turnsInDisruption = currentTurnNumber - self.disruptionCommencementTurn
            
            if turnsInDisruption > DISRUPTION_PHASE_DURATION_TURNS:
                self.currentOperationalMode = OperationalMode.PRODUCTION_MODE
                _emitDiagnostic(f"Returning to production mode at turn {currentTurnNumber}")
                
                for botId in teamBotIds:
                    self.botTaskPipelineState[botId] = {
                        'phase': TASK_STATE_AWAITING_ASSIGNMENT
                    }
                    self.activeOrderAssignments.pop(botId, None)
        
        # =====================================================================
        # EXECUTE BOT ACTIONS
        # =====================================================================
        for botId in teamBotIds:
            if self.currentOperationalMode == OperationalMode.DISRUPTION_MODE:
                self._executeDisruptionProtocol(controller, botId, ourTeam)
            else:
                self._executeProductionPipeline(controller, botId, ourTeam)
        
        # Log final statistics on last turn
        if DIAGNOSTIC_OUTPUT_ENABLED and currentTurnNumber == 499:
            _emitDiagnostic(
                f"=== FINAL STATS: cookwareSeized={self.cookwareItemsSeized}, "
                f"dishwareSeized={self.dishwareItemsSeized} ==="
            )


# =============================================================================
#  END OF FILE
# =============================================================================
#
# Development Notes (for future reference):
# -----------------------------------------
# - The BFS precomputation is O(n^2) in walkable cells but runs once per game
# - Considered A* but the precomputed tables give O(1) queries which matters more
# - Sabotage timing at turn 250 was determined empirically - earlier is riskier
# - The 60-turn disruption window balances damage vs lost production time
#
# Known Limitations:
# ------------------
# - No handling for dirty plates (could be added if washing becomes important)  
# - Single cooker assumption may break on multi-cooker maps
# - No bot coordination for parallel order processing
#
# Future Improvements (if we have time):
# --------------------------------------
# - Adaptive switch timing based on score differential
# - Priority queue for multi-bot order assignment
# - Counter reservation to prevent collisions
#
