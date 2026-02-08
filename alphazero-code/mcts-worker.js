/**
 * MCTS Web Worker for parallel simulations
 * Runs in a separate thread for better performance
 */

// ==================== GAME STATE (копия для воркера) ====================
class GameState {
    constructor() {
        this.pits = {
            white: [9, 9, 9, 9, 9, 9, 9, 9, 9],
            black: [9, 9, 9, 9, 9, 9, 9, 9, 9]
        };
        this.kazan = { white: 0, black: 0 };
        this.tuzdyk = { white: -1, black: -1 };
        this.currentPlayer = 'white';
    }
    
    static fromData(data) {
        const state = new GameState();
        state.pits = {
            white: [...data.pits.white],
            black: [...data.pits.black]
        };
        state.kazan = { ...data.kazan };
        state.tuzdyk = { ...data.tuzdyk };
        state.currentPlayer = data.currentPlayer;
        return state;
    }
    
    clone() {
        const state = new GameState();
        state.pits = {
            white: [...this.pits.white],
            black: [...this.pits.black]
        };
        state.kazan = { ...this.kazan };
        state.tuzdyk = { ...this.tuzdyk };
        state.currentPlayer = this.currentPlayer;
        return state;
    }
    
    getOpponent(player) {
        return player === 'white' ? 'black' : 'white';
    }
    
    getValidMoves(player) {
        const moves = [];
        const opponent = this.getOpponent(player);
        
        for (let i = 0; i < 9; i++) {
            // Пропускаем лунку, если противник объявил её түздық
            // Түздық противника находится на нашей стороне - мы не можем из неё играть
            if (this.tuzdyk[opponent] === i) continue;
            
            if (this.pits[player][i] > 0) moves.push(i);
        }
        return moves;
    }
    
    canCreateTuzdyk(player, pitIndex) {
        if (this.tuzdyk[player] !== -1) return false;
        if (pitIndex === 8) return false;
        const opponent = this.getOpponent(player);
        if (this.tuzdyk[opponent] === pitIndex) return false;
        return true;
    }
    
    makeMove(pitIndex) {
        const player = this.currentPlayer;
        const opponent = this.getOpponent(player);
        
        let stones = this.pits[player][pitIndex];
        if (stones === 0) return false;
        
        this.pits[player][pitIndex] = 0;
        
        let currentPit = pitIndex;
        let currentSide = player;
        
        if (stones === 1) {
            currentPit++;
            if (currentPit > 8) {
                currentPit = 0;
                currentSide = opponent;
            }
            
            // Проверяем, является ли лунка чьим-либо түздық
            const isWhiteTuzdyk = currentSide === 'black' && this.tuzdyk.white === currentPit;
            const isBlackTuzdyk = currentSide === 'white' && this.tuzdyk.black === currentPit;
            
            if (isWhiteTuzdyk) {
                this.kazan.white++;
            } else if (isBlackTuzdyk) {
                this.kazan.black++;
            } else {
                this.pits[currentSide][currentPit]++;
            }
        } else {
            this.pits[currentSide][currentPit]++;
            stones--;
            
            while (stones > 0) {
                currentPit++;
                if (currentPit > 8) {
                    currentPit = 0;
                    currentSide = currentSide === 'white' ? 'black' : 'white';
                }
                
                // Проверяем, является ли лунка чьим-либо түздық
                const isWhiteTuzdykLoop = currentSide === 'black' && this.tuzdyk.white === currentPit;
                const isBlackTuzdykLoop = currentSide === 'white' && this.tuzdyk.black === currentPit;
                
                if (isWhiteTuzdykLoop) {
                    this.kazan.white++;
                } else if (isBlackTuzdykLoop) {
                    this.kazan.black++;
                } else {
                    this.pits[currentSide][currentPit]++;
                }
                stones--;
            }
        }
        
        // Проверка на захват и создание түздық (только если это НЕ түздық)
        const isAnyTuzdyk = (currentSide === 'black' && this.tuzdyk.white === currentPit) ||
                           (currentSide === 'white' && this.tuzdyk.black === currentPit);
        
        if (currentSide === opponent && !isAnyTuzdyk) {
            const count = this.pits[opponent][currentPit];
            
            if (count === 3 && this.canCreateTuzdyk(player, currentPit)) {
                this.tuzdyk[player] = currentPit;
                this.kazan[player] += count;
                this.pits[opponent][currentPit] = 0;
            }
            else if (count % 2 === 0 && count > 0) {
                this.kazan[player] += count;
                this.pits[opponent][currentPit] = 0;
            }
        }
        
        this.currentPlayer = opponent;
        return true;
    }
    
    isGameOver() {
        if (this.kazan.white >= 82 || this.kazan.black >= 82) return true;
        const whiteEmpty = this.pits.white.every(p => p === 0);
        const blackEmpty = this.pits.black.every(p => p === 0);
        return whiteEmpty || blackEmpty;
    }
    
    getWinner() {
        if (this.kazan.white >= 82) return 'white';
        if (this.kazan.black >= 82) return 'black';
        if (this.kazan.white > this.kazan.black) return 'white';
        if (this.kazan.black > this.kazan.white) return 'black';
        return 'draw';
    }
}

// ==================== MCTS NODE ====================
class MCTSNode {
    constructor(state, parent = null, move = null) {
        this.state = state;
        this.parent = parent;
        this.move = move;
        this.children = [];
        this.wins = 0;
        this.visits = 0;
        this.untriedMoves = state.getValidMoves(state.currentPlayer);
    }
    
    isFullyExpanded() {
        return this.untriedMoves.length === 0;
    }
    
    hasChildren() {
        return this.children.length > 0;
    }
    
    getUCTValue(explorationWeight) {
        if (this.visits === 0) return Infinity;
        return (this.wins / this.visits) + 
               explorationWeight * Math.sqrt(Math.log(this.parent.visits) / this.visits);
    }
    
    selectChild(explorationWeight) {
        let best = null;
        let bestValue = -Infinity;
        
        for (const child of this.children) {
            const uct = child.getUCTValue(explorationWeight);
            if (uct > bestValue) {
                bestValue = uct;
                best = child;
            }
        }
        return best;
    }
    
    expand() {
        const move = this.untriedMoves.pop();
        const newState = this.state.clone();
        newState.makeMove(move);
        
        const childNode = new MCTSNode(newState, this, move);
        this.children.push(childNode);
        return childNode;
    }
    
    update(winner, aiPlayer) {
        this.visits++;
        if (winner === aiPlayer) {
            this.wins += 1;
        } else if (winner === 'draw') {
            this.wins += 0.5;
        }
    }
}

// ==================== WORKER MCTS ====================
function simulate(state, aiPlayer) {
    const simState = state.clone();
    let moveCount = 0;
    const maxMoves = 200;
    
    while (!simState.isGameOver() && moveCount < maxMoves) {
        const moves = simState.getValidMoves(simState.currentPlayer);
        if (moves.length === 0) break;
        
        let selectedMove;
        
        if (Math.random() < 0.7) {
            let bestScore = -Infinity;
            selectedMove = moves[0];
            
            for (const move of moves) {
                const testState = simState.clone();
                const kazanBefore = testState.kazan[testState.currentPlayer];
                testState.makeMove(move);
                const kazanAfter = testState.kazan[simState.currentPlayer];
                const score = kazanAfter - kazanBefore + Math.random() * 0.5;
                
                if (score > bestScore) {
                    bestScore = score;
                    selectedMove = move;
                }
            }
        } else {
            selectedMove = moves[Math.floor(Math.random() * moves.length)];
        }
        
        simState.makeMove(selectedMove);
        moveCount++;
    }
    
    return simState.getWinner();
}

function runMCTS(stateData, aiPlayer, simulations, timeLimit, explorationWeight = 1.41) {
    const rootState = GameState.fromData(stateData);
    const root = new MCTSNode(rootState.clone());
    const startTime = Date.now();
    let simulationsRun = 0;
    
    while (simulationsRun < simulations && (Date.now() - startTime) < timeLimit) {
        let node = root;
        
        // Selection
        while (node.isFullyExpanded() && node.hasChildren()) {
            node = node.selectChild(explorationWeight);
        }
        
        // Expansion
        if (!node.state.isGameOver() && !node.isFullyExpanded()) {
            node = node.expand();
        }
        
        // Simulation
        const winner = simulate(node.state, aiPlayer);
        
        // Backpropagation
        while (node !== null) {
            node.update(winner, aiPlayer);
            node = node.parent;
        }
        
        simulationsRun++;
        
        // Progress report every 5000 simulations
        if (simulationsRun % 5000 === 0) {
            self.postMessage({
                type: 'progress',
                simulations: simulationsRun,
                elapsed: Date.now() - startTime
            });
        }
    }
    
    // Find best move
    let bestMove = null;
    let bestVisits = -1;
    let bestWinRate = 0;
    
    for (const child of root.children) {
        if (child.visits > bestVisits) {
            bestVisits = child.visits;
            bestMove = child.move;
            bestWinRate = child.wins / child.visits;
        }
    }
    
    // Collect all move stats for logging
    const moveStats = root.children.map(c => ({
        move: c.move,
        visits: c.visits,
        wins: c.wins,
        winRate: c.visits > 0 ? (c.wins / c.visits) : 0
    }));
    
    return {
        bestMove,
        simulations: simulationsRun,
        elapsed: Date.now() - startTime,
        winRate: bestWinRate,
        moveStats
    };
}

// Worker message handler
self.onmessage = function(e) {
    const { stateData, aiPlayer, simulations, timeLimit } = e.data;
    
    const result = runMCTS(stateData, aiPlayer, simulations, timeLimit);
    
    self.postMessage({
        type: 'result',
        ...result
    });
};

