package io.github.akashprajapathi.tic_tac_toeai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import io.github.akashprajapathi.tic_tac_toeai.ui.screen.GameBoard
import io.github.akashprajapathi.tic_tac_toeai.ui.theme.TicTacToeAITheme
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            TicTacToeAITheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize()
                ) { innerPadding ->

                    MainScreen(
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun MainScreen(
    viewModel: BoardViewModel = viewModel(),
    modifier: Modifier = Modifier
) {
    val states by viewModel.states.collectAsState()
    val debug by viewModel.debug.collectAsState()

    var episode by remember {
        mutableIntStateOf(1)
    }

    val winner = viewModel.isWinState()
    val isTie = !states.any { it == 0 }
    val isEnd = winner != 0 || isTie

    LaunchedEffect(states) {
        delay(200)
        if (
            viewModel.getTurn() == 'X' &&
            !viewModel.isEnd()
        ) {
            viewModel.playAI()
        }
    }

    Column(
        modifier = modifier
            .padding(vertical = 16.dp)
            .fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Episode: $episode")

        Text(
            text = when {
                winner == 1 -> "X won"
                winner == -1 -> "O won"
                isTie -> "Tie"
                viewModel.getTurn() == 'X' -> "X's turn"
                else -> "O's turn"
            }
        )

        Spacer(Modifier.height(16.dp))

        /*
         * RL DEBUG
         */
        Text("AI Wins: ${debug.aiWins}")
        Text("AI Losses: ${debug.aiLosses}")
        Text("Draws: ${debug.draws}")

        Spacer(Modifier.height(8.dp))

        Text("Last Result: ${debug.lastResult}")
        Text("Reward: ${debug.lastReward}")
        Text("Loss: ${debug.lastLoss}")

        Spacer(Modifier.height(32.dp))

        if (!isEnd) {
            GameBoard(
                states = states,
                strokeWidth = 16f,
                cap = StrokeCap.Round,
                modifier = Modifier.size(300.dp)
            ) { row, col ->
                viewModel.move(row, col)
            }
        }

        if (isEnd) {
            Spacer(Modifier.height(32.dp))

            Button(
                onClick = {
                    viewModel.reset()
                    episode += 1
                }
            ) {
                Text("Play again!")
            }
        }
    }
}