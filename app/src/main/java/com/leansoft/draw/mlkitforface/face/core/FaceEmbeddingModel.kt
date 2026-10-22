package com.leansoft.draw.mlkitforface.face.core

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.InterpreterApi
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt
import androidx.core.graphics.scale

class FaceEmbeddingModel(context: Context) : AutoCloseable {
    private val modelBuffer: ByteBuffer = context.assets.open("mobile_face_net.tflite").use { input ->
            val bytes = input.readBytes()
            ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder()).apply { put(bytes); rewind() }
        }
    private val interpreter: InterpreterApi

    init {
        Tasks.await(TfLite.initialize(context))
        interpreter = InterpreterApi.create(
            modelBuffer,
            InterpreterApi.Options().setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
        )
        require(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1, INPUT_SIZE, INPUT_SIZE, 3))) {
            "MobileFaceNet input shape must be [1,112,112,3]"
        }
        require(interpreter.getOutputTensor(0).shape().contentEquals(intArrayOf(1, EMBEDDING_SIZE))) {
            "MobileFaceNet output shape must be [1,192]"
        }
    }

    @Synchronized
    fun embed(face: Bitmap): FloatArray {
        val normalized = face.scale(INPUT_SIZE, INPUT_SIZE)
        val input = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        normalized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        pixels.forEach { pixel ->
            input.putFloat(((pixel shr 16 and 0xff) / 128f) - 1f)
            input.putFloat(((pixel shr 8 and 0xff) / 128f) - 1f)
            input.putFloat(((pixel and 0xff) / 128f) - 1f)
        }
        input.rewind()
        val output = Array(1) { FloatArray(EMBEDDING_SIZE) }
        interpreter.run(input, output)
        if (normalized !== face) normalized.recycle()
        return normalize(output[0])
    }

    override fun close() = interpreter.close()

    companion object {
        const val INPUT_SIZE = 112
        const val EMBEDDING_SIZE = 192

        fun normalize(vector: FloatArray): FloatArray {
            var sum = 0.0
            vector.forEach { sum += it.toDouble() * it.toDouble() }
            val norm = sqrt(sum).toFloat().coerceAtLeast(1e-12f)
            return FloatArray(vector.size) { vector[it] / norm }
        }

        fun cosine(left: FloatArray, right: FloatArray): Float {
            if (left.size != right.size) return -1f
            var dot = 0f
            var leftNorm = 0f
            var rightNorm = 0f
            for (i in left.indices) {
                dot += left[i] * right[i]
                leftNorm += left[i] * left[i]
                rightNorm += right[i] * right[i]
            }
            return dot / (sqrt(leftNorm) * sqrt(rightNorm)).coerceAtLeast(1e-12f)
        }

        fun centroid(vectors: List<FloatArray>): FloatArray {
            require(vectors.isNotEmpty())
            val average = FloatArray(vectors.first().size)
            vectors.forEach { vector -> vector.indices.forEach { average[it] += vector[it] / vectors.size } }
            return normalize(average)
        }
    }
}
