package io.github.akashprajapathi.tic_tac_toeai.ui.screen

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput

@Composable
fun GameBoard(
    states: IntArray,
    strokeWidth: Float,
    modifier: Modifier = Modifier,
    cap: StrokeCap = Stroke.DefaultCap,
    onCellClick: (Int, Int) -> Unit
) {

    Box(
        modifier = modifier,
        contentAlignment = Alignment.Center
    ) {
        Canvas(
            modifier = Modifier
                .fillMaxSize(0.95f)
                .pointerInput(Unit) {
                    detectTapGestures { offset ->
                        val cellSize = size.width / 3

                        val column = (offset.x / cellSize).toInt()
                        val row = (offset.y / cellSize).toInt()

                        onCellClick(row, column)
                    }
                }
        ) {
            val cellWidth = size.width / 3
            val cellHeight = size.height / 3

            fun drawX(start: Offset) {
                drawLine(
                    color = Color.Black,
                    start = start,
                    end = start + Offset(cellWidth, cellHeight),
                    strokeWidth = strokeWidth,
                    cap = cap
                )

                drawLine(
                    color = Color.Black,
                    start = Offset(start.x + cellWidth, start.y),
                    end = start + Offset(0f, cellHeight),
                    strokeWidth = strokeWidth,
                    cap = cap
                )
            }

            fun drawO(start: Offset) {
                drawArc(
                    color = Color.Black,
                    startAngle = 0f,
                    sweepAngle = 360f,
                    topLeft = start,
                    size = Size(cellWidth, cellHeight),
                    useCenter = true,
                    style = Stroke(width = strokeWidth)
                )
            }

            for ((index, s) in states.withIndex()) {

                val row = index / 3
                val col = index % 3

                val start = getStartOffset(
                    row = row,
                    col = col,
                    size = size
                )

                if (s == 1) {
                    drawX(start)
                } else if (s == -1) {
                    drawO(start)
                }
            }

            // Draw horizontal lines
            drawLine(
                color = Color.Black,
                start = Offset(0f, cellHeight),
                end = Offset(size.width, cellHeight),
                strokeWidth = strokeWidth,
                cap = cap
            )

            drawLine(
                color = Color.Black,
                start = Offset(0f, 2 * cellHeight),
                end = Offset(size.width, 2 * cellHeight),
                strokeWidth = strokeWidth,
                cap = cap
            )

            // Draw vertical lines
            drawLine(
                color = Color.Black,
                start = Offset(cellWidth, 0f),
                end = Offset(cellWidth, size.height),
                strokeWidth = strokeWidth,
                cap = cap
            )

            drawLine(
                color = Color.Black,
                start = Offset(2 * cellWidth, 0f),
                end = Offset(2 * cellWidth, size.height),
                strokeWidth = strokeWidth,
                cap = cap
            )
        }
    }
}

fun getStartOffset(row: Int, col: Int, size: Size): Offset {
    val cellWidth = size.width / 3
    val cellHeight = size.height / 3

    return Offset(col * cellWidth, row * cellHeight)
}
