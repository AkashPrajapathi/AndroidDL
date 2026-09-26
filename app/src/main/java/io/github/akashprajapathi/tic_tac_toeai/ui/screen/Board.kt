package io.github.akashprajapathi.tic_tac_toeai.ui.screen

class Board {
    val states = IntArray(9)

    // X player
    val x = 1

    // O player
    val o = -1

    private var turn = 1

    private val winStates = listOf(
        // Horizontal
        listOf(0, 1, 2),
        listOf(3, 4, 5),
        listOf(6, 7, 8),

        // Vertical
        listOf(0, 3, 6),
        listOf(1, 4, 7),
        listOf(2, 5, 8),

        // Diagonal
        listOf(0, 4, 8),
        listOf(2, 4, 6)

    )

    fun setFirstTurn(c: Char) {
        turn = if (c == 'X') 1 else -1
    }

    fun getState(row: Int, col: Int): Int {

        return row * 3 + col
    }

    fun getTurn(): Char {
        return if (turn == 1) 'X' else 'O'
    }

    fun isWinState(): Int {

        if (winStates.any { state ->
                state.all { states[it] == x }
            }) {
            return x
        }

        if (winStates.any { state ->
                state.all { states[it] == o }
            }) {
            return o
        }

        return 0
    }

    fun play() {
        turn = if (turn == 1) {
            -1
        } else {
            1
        }
    }

    fun reset() {
        states.fill(0)
        turn = if (turn == 1) 1 else -1
    }

    fun isEnd(): Boolean {
        return !states.any { it == 0 } || isWinState() != 0
    }
}