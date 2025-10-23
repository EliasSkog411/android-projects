package com.example.back

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat


import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {
    val DIM_BATCH_SIZE = 1
    val DIM_PIXEL_SIZE = 3
    val IMAGE_SIZE_X = 512
    val IMAGE_SIZE_Y = 512
    private lateinit var interpreter: InterpreterApi

    fun loadImageFromDrawable(context: Context, drawableId: Int, targetWidth: Int, targetHeight: Int): TensorImage {
        // 1. Decode the drawable resource to a Bitmap
        val bitmap = BitmapFactory.decodeResource(context.resources, drawableId)

        // 2. Resize using ImageProcessor
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        var tensorImage = TensorImage(DataType.FLOAT32)


        return imageProcessor.process(TensorImage.fromBitmap(bitmap))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }


//        val initializeTask: Task<Void> by lazy { TfLite.initialize(this) }
        val initializeTask: Task<Void> by lazy { TfLite.initialize(this,
            TfLiteInitializationOptions.builder()
                .setEnableGpuDelegateSupport(true)
                .build()) }

        var modelBuffer = FileUtil.loadMappedFile(this, "very_good_cs.tflite")
        val gpu_text = findViewById<TextView>(R.id.gpu_text)
        var tensorImage: TensorImage = loadImageFromDrawable(this, R.drawable.test, 512, 512)

        // Create output buffer with expected shape (1, 512, 512, 1)
        val outputShape = intArrayOf(1, 512, 512, 1)
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)

        val fac = GpuDelegateFactory()
        val gpuDelegate = fac.create(RuntimeFlavor.APPLICATION)

        initializeTask.addOnSuccessListener {
            val interpreterOption = InterpreterApi.Options()
                .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
                //.addDelegateFactory(GpuDelegateFactory())

            interpreter = InterpreterApi.create(
                modelBuffer,
                interpreterOption
            )


            val input = FloatBuffer.allocate(interpreter.getInputTensor(0).numElements())
            var test = fillInputBufferFromBitmap(tensorImage.bitmap, input)
            val output = FloatBuffer.allocate(interpreter.getOutputTensor(0).numElements())


            val startTime = SystemClock.uptimeMillis()
            interpreter.run(input, output)
            val time = SystemClock.uptimeMillis() - startTime

            gpu_text.text = ("GPU inference: " + time.toString())

        }.addOnFailureListener { e ->
                Log.e("Interpreter", "Cannot initialize interpreter", e)
            }


    }

    fun fillInputBufferFromBitmap(bitmap: Bitmap, inputBuffer: FloatBuffer) {
        // Resize bitmap to 512x512
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 512, 512, true)

        val pixels = IntArray(512 * 512)
        resizedBitmap.getPixels(pixels, 0, 512, 0, 0, 512, 512)

        inputBuffer.rewind()

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f

            inputBuffer.put(r)
            inputBuffer.put(g)
            inputBuffer.put(b)
        }

        inputBuffer.rewind()
    }

    private fun bitmapToFloatTensorBuffer(bitmap: Any) {}
}