package com.leansoft.draw.mlkitforface

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.net.Uri
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.leansoft.draw.mlkitforface.databinding.ActivityFaceDemoBinding
import java.io.IOException
import kotlin.math.roundToInt

class FaceDemoActivity : AppCompatActivity() {
    private lateinit var binding: ActivityFaceDemoBinding

    //use faceDetector
    private val faceDetector by lazy {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setMinFaceSize(0.15f)
            .build()

        FaceDetection.getClient(options)
    }


    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            uri?.let {
                binding.ivOriginal.setImageURI(it)
                analyzeImage(it)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityFaceDemoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        register()
    }

    private fun register() {
        binding.btnPickImage.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }
    }

    private fun analyzeImage(uri: Uri){
        binding.tvResult.text="Dang phan tich"
        val bitmap=contentResolver.openInputStream(uri)
            ?.use { BitmapFactory.decodeStream(it) }
        if(bitmap==null){
            binding.tvResult.text="Khong the doc anh"
            return
        }
        val inputImage = try {
            InputImage.fromFilePath(this, uri)
        } catch (e: IOException) {
            binding.tvResult.text = "Không tạo được InputImage"
            return
        }

        faceDetector.process(inputImage)
            .addOnSuccessListener { faces->
                handleFaces(faces,bitmap)
            }
            .addOnFailureListener {error->
                binding.tvResult.text="Loi: ${error.message}"
            }
    }

    private fun handleFaces(
        faces:List<Face>,
        bitmap: Bitmap
    ){
        if(faces.isEmpty()){
            binding.tvResult.text="Khong tim thay mat"
            return
        }

        if(faces.size>1){
            binding.tvResult.text="Tim thay nhieu mat"
            return
        }
        val face = faces.first()
        val bounds = face.boundingBox

        val faceCrop = cropFace(
            bitmap = bitmap,
            bounds = bounds
        )

        binding.ivFaceCrop.setImageBitmap(faceCrop)

        binding.tvResult.text = buildString {
            appendLine("Phát hiện: 1 khuôn mặt")
            appendLine("Bounding box: $bounds")
            appendLine(
                "Yaw: ${"%.2f".format(face.headEulerAngleY)}°"
            )
            appendLine(
                "Pitch: ${"%.2f".format(face.headEulerAngleX)}°"
            )
            appendLine(
                "Roll: ${"%.2f".format(face.headEulerAngleZ)}°"
            )
            appendLine(
                "Face size: ${bounds.width()} x ${bounds.height()}"
            )
        }
    }

    private fun cropFace(
        bitmap: Bitmap,
        bounds: Rect,
        marginRatio: Float = 0.25f
    ): Bitmap {
        val marginX =
            (bounds.width() * marginRatio).roundToInt()

        val marginY =
            (bounds.height() * marginRatio).roundToInt()

        val left = (bounds.left - marginX)
            .coerceAtLeast(0)

        val top = (bounds.top - marginY)
            .coerceAtLeast(0)

        val right = (bounds.right + marginX)
            .coerceAtMost(bitmap.width)

        val bottom = (bounds.bottom + marginY)
            .coerceAtMost(bitmap.height)

        return Bitmap.createBitmap(
            bitmap,
            left,
            top,
            right - left,
            bottom - top
        )
    }

    override fun onDestroy() {
        faceDetector.close()
        super.onDestroy()
    }
}