package com.example.back2

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.os.SystemClock
import android.provider.ContactsContract.Data
import android.util.Log
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageProxy
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.acceleration.BenchmarkResult
import com.google.android.gms.tflite.acceleration.CustomValidationConfig
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ColorSpaceType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.ImageProperties
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import androidx.core.graphics.scale


/*
*     var inputshapetext = interpreter.getInputTensor(0).shapeSignature().joinToString(prefix = "(", postfix = ")", separator = ", ")

            Log.d("INPUT SHAPE", inputshapetext.toString())

            val startTime = SystemClock.uptimeMillis()
            val input = loadImageFromDrawableToBuffer(this,R.drawable.tre,512,512)//tensorImage.buffer.asFloatBuffer()
            val output_byte_buffer = ByteBuffer.allocate(512*512)
            input.rewind()
            val bbbiittmap = createBitmapFromByteBuffer(input, 512, 512)
            imageView.setImageBitmap(bbbiittmap)
            input.rewind()

            // runs model

            interpreter.run(input, output_byte_buffer)
            val time = SystemClock.uptimeMillis() - startTime
            a.text = ("Inference: " + time.toString())


            val imageProperties = ImageProperties.builder()
                .setColorSpaceType(ColorSpaceType.GRAYSCALE)
                .setWidth(512)
                .setHeight(512)
                .build()

            val tempTensorBuffer = TensorBuffer.createDynamic(DataType.UINT8)
            tempTensorBuffer.loadBuffer(output_byte_buffer,  intArrayOf(1, 512, 512, 1))

            var modelImage = TensorImage(DataType.UINT8)
            modelImage.load(tempTensorBuffer, ColorSpaceType.GRAYSCALE)
* */

class LogVision private constructor() {
    // Private constructor prevents direct instantiation

    init {
        // Any initialization logic
    }

    companion object {
        fun createAsync(): Task<LogVision> {
            return Tasks.call(Executors.newSingleThreadExecutor()) {
                // Perform any setup or heavy init if needed
                LogVision()
            }
        }
    }
}


/*
    PipeLine owns the model
 */
class PipeLine(
    private val context: Context,
    private val modelSize: Int
) {
    public var isSuccessful: Boolean

    private lateinit var interpreter: InterpreterApi
    private val outbutBuffer: ByteBuffer
    private val inputBuffer: ByteBuffer

    private val modelBuffer: java.nio.MappedByteBuffer



    init {
        isSuccessful = false
        outbutBuffer = ByteBuffer.allocateDirect(512*512)
        inputBuffer =  ByteBuffer.allocateDirect(512*512*4)
        modelBuffer = FileUtil.loadMappedFile(context, "quant_uint8.tflite")

        // Checks if gpu is available
        val gpuAvailableTask: Task<Boolean> = TfLiteGpu.isGpuDelegateAvailable(context)
        Tasks.await(gpuAvailableTask)
        val useGpu = gpuAvailableTask.result

        // Preps the Tflite environment
        val interpreterTask = TfLite.initialize(
            context,
            TfLiteInitializationOptions.builder()
                .setEnableGpuDelegateSupport(useGpu)
                .build()
        )

        Tasks.await(interpreterTask)


        if(interpreterTask.isSuccessful) {
            isSuccessful = true
            var options = InterpreterApi.Options()
                .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
                .addDelegateFactory(GpuDelegateFactory())

            if(!useGpu) {
                options = InterpreterApi.Options()
                    .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
                    .setUseXNNPACK(true)
                    .setNumThreads(5)
            }

            Log.d("GPU STATUS", if (useGpu) "USING GPU" else "NOT USING GPU")

            // sets up inference
            interpreter = InterpreterApi.create(
                modelBuffer,
                options
            )
        }

    }

    fun runModel(bitmap: Bitmap): Bitmap? {
        val resizedBitmap = bitmap.scale(modelSize, modelSize)

        // reset the bufffers
        inputBuffer.rewind()
        outbutBuffer.rewind()

        resizedBitmap.copyPixelsToBuffer(inputBuffer)
        inputBuffer.rewind()
        interpreter.run(inputBuffer, outbutBuffer)


        val startTime = SystemClock.uptimeMillis()
        val imageProperties = ImageProperties.builder()
            .setColorSpaceType(ColorSpaceType.GRAYSCALE)
            .setWidth(512)
            .setHeight(512)
            .build()

        var modelImage = TensorImage(DataType.UINT8)
        val tempTensorBuffer = TensorBuffer.createDynamic(DataType.UINT8)
        tempTensorBuffer.loadBuffer(outbutBuffer,  intArrayOf(1, 512, 512, 1))
        modelImage.load(tempTensorBuffer, ColorSpaceType.GRAYSCALE)

        val time = SystemClock.uptimeMillis() - startTime
        Log.d("DRAW_DEBUG", "TIME: " + time.toString())


        return null
    }
}

data class LogVisionOutput(
    var middlePoints: MutableList<Float>,
    var contour: MutableList<Float>,
    var modelOutput: Bitmap
)

/*ImageProxy
    Runs the model
 */
class LogVision(
    private val context: Context,
) {
    val pipeLine: PipeLine

    init {
        pipeLine = PipeLine(context, 512)
    }

    fun performInference(bitmap: Bitmap) : Bitmap? {
        if(!pipeLine.isSuccessful) {
            return null
        }

        return pipeLine.runModel(bitmap)
    }
}

/// https://ai.google.dev/edge/litert/android/java

class MainActivity : AppCompatActivity() {
    val DIM_BATCH_SIZE = 1
    val DIM_PIXEL_SIZE = 3
    val IMAGE_SIZE_X = 512
    val IMAGE_SIZE_Y = 512
    private lateinit var interpreter: InterpreterApi

    fun printFloatBufferToLogcat(buffer: FloatBuffer, tag: String = "FloatBuffer") {
        buffer.rewind()  // Make sure we're starting from position 0

        val builder = StringBuilder()

        var count = 0
        while (buffer.hasRemaining()) {
            builder.append(buffer.get().toString()).append(", ")

            // Optional: break into multiple lines for readability
            if (++count % 45 == 0) {
                Log.d(tag, builder.toString())
                builder.clear()
            }
        }

        // Print any remaining values
        if (builder.isNotEmpty()) {
            Log.d(tag, builder.toString())
        }

        buffer.rewind()  // Reset buffer if it needs to be reused
    }

    fun loadImageFromDrawableToBuffer(context: Context, drawableId: Int, targetWidth: Int, targetHeight: Int): ByteBuffer {
        val bitmap = BitmapFactory.decodeResource(context.resources, drawableId)
        val startTime = SystemClock.uptimeMillis()
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)

        resizedBitmap.isPremultiplied
        // creates bytebuffer from bitmap
        var byteBuffer = ByteBuffer.allocateDirect(512*512*4)
        resizedBitmap.copyPixelsToBuffer(byteBuffer)
        byteBuffer.rewind()

        val time = SystemClock.uptimeMillis() - startTime
        Log.d("DRAW_DEBUG", "TIME: " + time.toString())
        Log.d("DRAW_DEBUG", "BUFFER REMAINING: " + byteBuffer.remaining())
        Log.d("DRAW_DEBUG", "BUFFER POS: " + byteBuffer.position())
        Log.d("DRAW_DEBUG", "IS MULTIPLE: " + resizedBitmap.isPremultiplied.toString())

        Log.d("TEST","THE ARRAY: " +  byteBuffer.hasArray().toString())

        byteBuffer.rewind()


        return byteBuffer
    }

    fun loadImageFromDrawableToBuffer2(context: Context, drawableId: Int, targetWidth: Int, targetHeight: Int): FloatBuffer {
        val bitmap = BitmapFactory.decodeResource(context.resources, drawableId)
        val startTime = SystemClock.uptimeMillis()
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)

        // creates bytebuffer from bitmap
        var byteBuffer = ByteBuffer.allocateDirect(512*512*4)
        resizedBitmap.copyPixelsToBuffer(byteBuffer)
        byteBuffer.rewind()

        // creates floatbuffer from bytebuffer
        var outputFloatBuffer = byteBuffer.asFloatBuffer()
        outputFloatBuffer.rewind()

        val time = SystemClock.uptimeMillis() - startTime
        Log.d("DRAW_DEBUG", "TIME: " + time.toString())
        Log.d("DRAW_DEBUG", "BUFFER REMAINING: " + outputFloatBuffer.remaining())
        outputFloatBuffer.rewind()

        Log.d("TEST","THE ARRAY: " +  outputFloatBuffer.hasArray().toString())
        outputFloatBuffer.rewind()


        return outputFloatBuffer
    }

    fun loadImageFromDrawable(context: Context, drawableId: Int, targetWidth: Int, targetHeight: Int): TensorImage {
        // 1. Decode the drawable resource to a Bitmap
        val bitmap = BitmapFactory.decodeResource(context.resources, drawableId)



        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)

        TensorImage.fromBitmap(resizedBitmap)
        var tensor = TensorImage.fromBitmap(resizedBitmap)


        TensorImage.fromBitmap(resizedBitmap)

        val startTime = SystemClock.uptimeMillis()

        val floatArray = tensor.tensorBuffer.floatArray



        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(floatArray, intArrayOf(1, 512, 512, 3))


        val time = SystemClock.uptimeMillis() - startTime


        return tensorImage
    }

    fun loadImageFromDrawable2(context: Context, drawableId: Int, targetWidth: Int, targetHeight: Int): TensorImage {
        // 1. Decode the drawable resource to a Bitmap
        val bitmap = BitmapFactory.decodeResource(context.resources, drawableId)
        val startTime = SystemClock.uptimeMillis()



        // 2. Resize using ImageProcessor
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(targetHeight, targetWidth, ResizeOp.ResizeMethod.BILINEAR))
            .build()


        val tee = TensorImage.fromBitmap(bitmap)

        var floatArray = imageProcessor.process(tee).tensorBuffer.floatArray



        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(floatArray, intArrayOf(1, 512, 512, 3))


        val time = SystemClock.uptimeMillis() - startTime
        Log.d("typdsasdasdasdasdasdasdasdasde", time.toString())
        Log.d("typdsasdasdasdasdasdasdasdasde", tensorImage.dataType.toString())
        Log.d("typdsasdasdasdasdasdasdasdasde", tensorImage.tensorBuffer.floatArray.joinToString() )

        return tensorImage
    }

    fun createBitmapFromByteBuffer(buffer: ByteBuffer, width: Int, height: Int): Bitmap {
        buffer.rewind()  // Ensure position is at start

        // Create an empty mutable bitmap
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Copy buffer pixels into bitmap
        bitmap.copyPixelsFromBuffer(buffer)

        return bitmap
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

        var useGpu = false
        val useGpuTask: Task<Boolean> = TfLiteGpu.isGpuDelegateAvailable(this)
        val interpreterTask: Task<Void> = useGpuTask.onSuccessTask { gpuAvailable ->
            useGpu = gpuAvailable
            TfLite.initialize(
                this,
                TfLiteInitializationOptions.builder()
                    .setEnableGpuDelegateSupport(useGpu)
                    .build()
            )
        }

        // gets views
        val a = findViewById<TextView>(R.id.a)
        val b = findViewById<TextView>(R.id.b)
        val c = findViewById<TextView>(R.id.c)
        val imageView = findViewById<ImageView>(R.id.imageView)
        val modelView = findViewById<ImageView>(R.id.modelView)


        // loads model and input
        var modelBuffer = FileUtil.loadMappedFile(this, "quant_uint8.tflite")
        var tensorImage: TensorImage = loadImageFromDrawable(this, R.drawable.tre, 512, 512)


        interpreterTask.addOnSuccessListener {
            var options = InterpreterApi.Options()
                .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
                .addDelegateFactory(GpuDelegateFactory())

            if(!useGpu) {
                b.text = "NOT USING GPU"
                options = InterpreterApi.Options()
                    .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
                    .setUseXNNPACK(true)
                    .setNumThreads(5)
            } else {
                b.text = "USING GPU"
            }

            // sets up inference
            interpreter = InterpreterApi.create(
                modelBuffer,
                options
            )

            var inputshapetext = interpreter.getInputTensor(0).shapeSignature().joinToString(prefix = "(", postfix = ")", separator = ", ")

            Log.d("INPUT SHAPE", inputshapetext.toString())

            val startTime = SystemClock.uptimeMillis()
            val input = loadImageFromDrawableToBuffer(this,R.drawable.tre,512,512)//tensorImage.buffer.asFloatBuffer()
            val output_byte_buffer = ByteBuffer.allocate(512*512)
            input.rewind()
            val bbbiittmap = createBitmapFromByteBuffer(input, 512, 512)
            imageView.setImageBitmap(bbbiittmap)
            input.rewind()

            // runs model

            interpreter.run(input, output_byte_buffer)
            val time = SystemClock.uptimeMillis() - startTime
            a.text = ("Inference: " + time.toString())


            val imageProperties = ImageProperties.builder()
                .setColorSpaceType(ColorSpaceType.GRAYSCALE)
                .setWidth(512)
                .setHeight(512)
                .build()

            var modelImage = TensorImage(DataType.UINT8)
            val tempTensorBuffer = TensorBuffer.createDynamic(DataType.UINT8)
            tempTensorBuffer.loadBuffer(output_byte_buffer,  intArrayOf(1, 512, 512, 1))
            modelImage.load(tempTensorBuffer, ColorSpaceType.GRAYSCALE)
            Log.d("FUCKING OUTPUT", output_byte_buffer.array().joinToString())
            modelView.setImageBitmap(modelImage.bitmap)

            /*
            val asd = interpreter.getOutputTensor(0)
            val asdd =asd.asReadOnlyBuffer()
            val output_tensor_image = TensorImage(DataType.UINT8)
            val buf = TensorBuffer.createDynamic(DataType.UINT8)

            buf.loadBuffer(output_byte_buffer.asReadOnlyBuffer(), asd.shapeSignature())

            output_tensor_image.load(buf, ColorSpaceType.GRAYSCALE)
            Log.d("FileDump", asd.numBytes().toString())


            Log.d("ByteArrayCheck", "Complete")

            if (output_tensor_image.bitmap == null) {
                o1.text = "FUCKING NULLL"
            } else {
                img.setImageBitmap(output_tensor_image.bitmap)
            }
                        val output_tensor_image = TensorImage(DataType.FLOAT32)


                        img.setImageBitmap(output_tensor_image.bitmap)*/

        }.addOnFailureListener { e ->
            Log.e("Interpreter", "Cannot initialize interpreter", e)
        }


    }

    fun createTensorBufferFromByteBuffer(buffer: ByteBuffer): TensorBuffer {
        val shape = intArrayOf(1, 512, 512, 1)
        val tensorBuffer = TensorBuffer.createFixedSize(shape, DataType.UINT8)
        tensorBuffer.loadBuffer(buffer, shape)
        return tensorBuffer
    }

    fun fillInputBufferFromBitmap(bitmap: Bitmap, inputBuffer: FloatBuffer): FloatBuffer {
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

        return inputBuffer
    }


    private fun bitmapToFloatTensorBuffer(bitmap: Any) {}
}
