/**
 * AlphaZero Neural Network inference for Тоғызқұмалақ
 * Uses ONNX Runtime Web for browser inference
 */

class AlphaZeroNN {
    constructor() {
        this.session = null;
        this.metadata = null;
    }
    
    async load(modelPath = 'model.onnx', metadataPath = 'metadata.json') {
        // Load ONNX Runtime
        if (typeof ort === 'undefined') {
            throw new Error('ONNX Runtime Web not loaded. Add: <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>');
        }
        
        // Load model
        this.session = await ort.InferenceSession.create(modelPath);
        console.log('[AlphaZeroNN] Model loaded');
        
        // Load metadata
        const metaResponse = await fetch(metadataPath);
        this.metadata = await metaResponse.json();
        console.log('[AlphaZeroNN] Metadata loaded:', this.metadata);
    }
    
    /**
     * Encode game state for neural network
     * @param {Object} state - Game state {pits, kazan, tuzdyk, currentPlayer}
     * @returns {Float32Array} - Encoded state tensor
     */
    encodeState(state) {
        const PIT_NORM = 50.0;
        const KAZAN_NORM = 82.0;
        
        const player = state.currentPlayer === 'white' ? 0 : 1;
        const opponent = 1 - player;
        
        const playerPits = player === 0 ? state.pits.white : state.pits.black;
        const opponentPits = player === 0 ? state.pits.black : state.pits.white;
        const playerKazan = player === 0 ? state.kazan.white : state.kazan.black;
        const opponentKazan = player === 0 ? state.kazan.black : state.kazan.white;
        const playerTuzdyk = player === 0 ? state.tuzdyk.white : state.tuzdyk.black;
        const opponentTuzdyk = player === 0 ? state.tuzdyk.black : state.tuzdyk.white;
        
        // 7 channels x 9 positions
        const tensor = new Float32Array(7 * 9);
        
        // Channel 0: Player's pits (normalized)
        for (let i = 0; i < 9; i++) {
            tensor[0 * 9 + i] = playerPits[i] / PIT_NORM;
        }
        
        // Channel 1: Opponent's pits (normalized)
        for (let i = 0; i < 9; i++) {
            tensor[1 * 9 + i] = opponentPits[i] / PIT_NORM;
        }
        
        // Channel 2: Player's kazan (broadcast)
        for (let i = 0; i < 9; i++) {
            tensor[2 * 9 + i] = playerKazan / KAZAN_NORM;
        }
        
        // Channel 3: Opponent's kazan (broadcast)
        for (let i = 0; i < 9; i++) {
            tensor[3 * 9 + i] = opponentKazan / KAZAN_NORM;
        }
        
        // Channel 4: Player's tuzdyk (one-hot)
        if (playerTuzdyk >= 0) {
            tensor[4 * 9 + playerTuzdyk] = 1.0;
        }
        
        // Channel 5: Opponent's tuzdyk (one-hot)
        if (opponentTuzdyk >= 0) {
            tensor[5 * 9 + opponentTuzdyk] = 1.0;
        }
        
        // Channel 6: Current player indicator (white=1, black=0)
        const playerIndicator = player === 0 ? 1.0 : 0.0;
        for (let i = 0; i < 9; i++) {
            tensor[6 * 9 + i] = playerIndicator;
        }
        
        return tensor;
    }
    
    /**
     * Run inference on game state
     * @param {Object} state - Game state
     * @returns {Object} - {policy: Float32Array, value: number}
     */
    async predict(state) {
        if (!this.session) {
            throw new Error('Model not loaded. Call load() first.');
        }
        
        // Encode state
        const encoded = this.encodeState(state);
        
        // Create tensor [1, 7, 9]
        const inputTensor = new ort.Tensor('float32', encoded, [1, 7, 9]);
        
        // Run inference
        const results = await this.session.run({ state: inputTensor });
        
        // Extract outputs
        const logPolicy = results.policy.data;
        const value = results.value.data[0];
        
        // Convert log policy to probabilities
        const policy = new Float32Array(9);
        let maxLogP = -Infinity;
        for (let i = 0; i < 9; i++) {
            if (logPolicy[i] > maxLogP) maxLogP = logPolicy[i];
        }
        
        let sumExp = 0;
        for (let i = 0; i < 9; i++) {
            policy[i] = Math.exp(logPolicy[i] - maxLogP);
            sumExp += policy[i];
        }
        for (let i = 0; i < 9; i++) {
            policy[i] /= sumExp;
        }
        
        return { policy, value };
    }
    
    /**
     * Get best move from policy
     * @param {Float32Array} policy - Policy distribution
     * @param {Array<number>} validMoves - List of valid move indices
     * @returns {number} - Best move index
     */
    getBestMove(policy, validMoves) {
        let bestMove = validMoves[0];
        let bestProb = -Infinity;
        
        for (const move of validMoves) {
            if (policy[move] > bestProb) {
                bestProb = policy[move];
                bestMove = move;
            }
        }
        
        return bestMove;
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AlphaZeroNN };
}
