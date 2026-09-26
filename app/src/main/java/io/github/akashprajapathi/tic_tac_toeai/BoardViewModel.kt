package io.github.akashprajapathi.tic_tac_toeai

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import io.github.akashprajapathi.androiddl.io.ModelIO
import io.github.akashprajapathi.androiddl.optim.SGD
import io.github.akashprajapathi.tic_tac_toeai.rl.Episode
import io.github.akashprajapathi.tic_tac_toeai.rl.PolicyAgent
import io.github.akashprajapathi.tic_tac_toeai.rl.PolicyNet
import io.github.akashprajapathi.tic_tac_toeai.rl.ValueNet
import io.github.akashprajapathi.tic_tac_toeai.ui.screen.Board
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

data class DebugState(
    val aiWins: Int = 0,
    val aiLosses: Int = 0,
    val draws: Int = 0,
    val lastResult: String = "-",
    val lastReward: Float = 0f,
    val lastLoss: Float = 0f
)

// Changed to AndroidViewModel so we have access to context (`getApplication()`) for file storage on startup
class BoardViewModel(application: Application) : AndroidViewModel(application) {

    private val POLICY_MODEL_FILE = "tictactoe_policy.bin"
    private val VALUE_MODEL_FILE = "tictactoe_value.bin"

    private val board = Board()

    private val policy = PolicyNet()

    private val valueNet = ValueNet()

    private val agent = PolicyAgent(
        policy = policy,
        valueNet = valueNet
    )

    private val episode = Episode()

    private val policyOptimizer = SGD(
        parameters = policy.parameters(),
        lr = 0.01f,
        momentum = 0.09f
    )

    private val valueOptimizer = SGD(
        parameters = valueNet.parameters(),
        lr = 0.01f,
        momentum = 0.09f
    )

    private var episodeCount = 1

    /*
     * Board state exposed to Compose
     */
    private val _states =
        MutableStateFlow(board.states.copyOf())

    val states =
        _states.asStateFlow()

    /*
     * RL debug state exposed to Compose
     */
    private val _debug =
        MutableStateFlow(DebugState())

    val debug =
        _debug.asStateFlow()

    init {
        // Automatically load model weights & optimizer state when app/viewmodel opens
        loadModel()
    }

    fun move(row: Int, col: Int) {
        val index = board.getState(row, col)

        if (board.states[index] != 0) return

        board.states[index] =
            if (board.getTurn() == 'X') {
                board.x
            } else {
                board.o
            }

        board.play()

        _states.value = board.states.copyOf()

        if (board.isEnd()) {
            trainEpisode()
        }
    }

    fun playAI() {
        if (board.isEnd()) return
        if (board.getTurn() != 'X') return

        val result = agent.selectAction(states = board.states)

        episode.add(
            logProb = result.logProb,
            value = result.value
        )

        val action = result.action

        move(
            row = action / 3,
            col = action % 3
        )
    }

    private fun trainEpisode() {

        val winner = board.isWinState()

        val reward =
            when (winner) {
                board.x -> 1f
                board.o -> -1f
                else -> 0f
            }

        if (episode.isEmpty()) return

        // The final AI action receives the terminal reward.
        episode.setLastReward(reward)

        val gamma = 0.99f

        val actorLoss = episode.actorLoss(gamma)
        val criticLoss = episode.criticLoss(gamma)

        policyOptimizer.zeroGrad()
        actorLoss.backward()
        policyOptimizer.step()

        valueOptimizer.zeroGrad()
        criticLoss.backward()
        valueOptimizer.step()

        val result =
            when (reward) {
                1f -> "X WON"
                -1f -> "O WON"
                else -> "DRAW"
            }

        val current = _debug.value

        _debug.value =
            current.copy(
                aiWins =
                    current.aiWins +
                            if (reward == 1f) 1 else 0,

                aiLosses =
                    current.aiLosses +
                            if (reward == -1f) 1 else 0,

                draws =
                    current.draws +
                            if (reward == 0f) 1 else 0,

                lastResult = result,
                lastReward = reward,
                lastLoss = actorLoss.data[0]
            )

        episode.clear()

        if (episodeCount % 5 == 0) {
            saveModel()
        }

        episodeCount++
    }

    fun getTurn(): Char = board.getTurn()

    fun isWinState(): Int = board.isWinState()

    fun isEnd(): Boolean = board.isEnd()

    fun reset() {
        board.reset()
        episode.clear()
        _states.value = board.states.copyOf()
    }

    private fun saveModel() {

        val policyFile =
            File(
                getApplication<Application>().filesDir,
                POLICY_MODEL_FILE
            )

        ModelIO.saveCheckpoint(
            module = policy,
            optimizer = policyOptimizer,
            filePath = policyFile.absolutePath
        )

        val valueFile =
            File(
                getApplication<Application>().filesDir,
                VALUE_MODEL_FILE
            )

        ModelIO.saveCheckpoint(
            module = valueNet,
            optimizer = valueOptimizer,
            filePath = valueFile.absolutePath
        )
    }

    private fun loadModel() {

        val policyFile =
            File(
                getApplication<Application>().filesDir,
                POLICY_MODEL_FILE
            )

        if (policyFile.exists()) {
            ModelIO.loadCheckpoint(
                module = policy,
                optimizer = policyOptimizer,
                filePath = policyFile.absolutePath
            )
        }

        val valueFile =
            File(
                getApplication<Application>().filesDir,
                VALUE_MODEL_FILE
            )

        if (valueFile.exists()) {
            ModelIO.loadCheckpoint(
                module = valueNet,
                optimizer = valueOptimizer,
                filePath = valueFile.absolutePath
            )
        }
    }
}