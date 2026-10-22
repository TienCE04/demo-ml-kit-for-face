package com.leansoft.draw.mlkitforface.face.core

import android.content.Context
import android.graphics.Bitmap
import androidx.core.graphics.scale
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** Optional FaceNet comparator. It is kept separate from the primary MobileFaceNet model. */
class FaceNetEmbeddingModel(context: Context) : AutoCloseable {
    private val modelBuffer = context.assets.open("facenet_512_int_quantized.tflite").use { input ->
        val bytes = input.readBytes()
        ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }
    }
    private val interpreter: InterpreterApi
    private val inputTensor: org.tensorflow.lite.Tensor
    private val outputTensor: org.tensorflow.lite.Tensor
    private val inputSize: Int

    init {
        Tasks.await(TfLite.initialize(context))
        interpreter = InterpreterApi.create(
            modelBuffer,
            InterpreterApi.Options().setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
        )
        inputTensor = interpreter.getInputTensor(0)
        outputTensor = interpreter.getOutputTensor(0)
        val shape = inputTensor.shape()
        require(shape.size == 4 && shape[0] == 1 && shape[3] == 3) { "FaceNet input must be NHWC RGB" }
        inputSize = shape[1]
        require(outputTensor.shape().size == 2 && outputTensor.shape()[0] == 1) { "FaceNet output must be [1,N]" }
    }

    @Synchronized
    fun embed(face: Bitmap): FloatArray {
        val resized = face.scale(inputSize, inputSize)
        val input = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())
        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        pixels.forEach { pixel ->
            intArrayOf(pixel shr 16 and 0xff, pixel shr 8 and 0xff, pixel and 0xff)
                .forEach { channel -> putInput(input, channel / 128f - 1f) }
        }
        input.rewind()
        val output = readOutput(input)
        if (resized !== face) resized.recycle()
        return normalize(output)
    }

    private fun putInput(buffer: ByteBuffer, value: Float) {
        when (inputTensor.dataType()) {
            DataType.FLOAT32 -> buffer.putFloat(value)
            DataType.INT8, DataType.UINT8 -> {
                val params = inputTensor.quantizationParams()
                val quantized = (value / params.scale + params.zeroPoint).roundToInt()
                val lower = if (inputTensor.dataType() == DataType.INT8) -128 else 0
                buffer.put(quantized.coerceIn(lower, 255).toByte())
            }
            else -> error("Unsupported FaceNet input type: ${inputTensor.dataType()}")
        }
    }

    private fun readOutput(input: ByteBuffer): FloatArray {
        val length = outputTensor.shape()[1]
        return when (outputTensor.dataType()) {
            DataType.FLOAT32 -> Array(1) { FloatArray(length) }.also { interpreter.run(input, it) }[0]
            DataType.INT8, DataType.UINT8 -> {
                val buffer = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
                interpreter.run(input, buffer)
                buffer.rewind()
                val params = outputTensor.quantizationParams()
                FloatArray(length) {
                    val raw = if (outputTensor.dataType() == DataType.INT8) buffer.get().toInt() else buffer.get().toInt() and 0xff
                    (raw - params.zeroPoint) * params.scale
                }
            }
            else -> error("Unsupported FaceNet output type: ${outputTensor.dataType()}")
        }
    }

    override fun close() = interpreter.close()

    private fun normalize(vector: FloatArray): FloatArray {
        val norm = sqrt(vector.sumOf { it.toDouble() * it.toDouble() }).toFloat().coerceAtLeast(1e-12f)
        return FloatArray(vector.size) { vector[it] / norm }
    }
}
