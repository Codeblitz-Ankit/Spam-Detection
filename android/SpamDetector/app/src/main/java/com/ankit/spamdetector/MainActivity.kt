package com.ankit.spamdetector

import android.Manifest
import android.content.pm.PackageManager
import android.database.Cursor
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.provider.Telephony
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

// ── Colors ───────────────────────────────────────────
private val Background     = Color(0xFF0F0F0F)
private val Surface1       = Color(0xFF1A1A1A)
private val Surface2       = Color(0xFF242424)
private val OnSurface      = Color(0xFFE8E8E8)
private val OnSurfaceMuted = Color(0xFF8A8A8A)
private val Accent         = Color(0xFF5C6BC0)
private val SpamColor      = Color(0xFFE53935)
private val SpamSurface    = Color(0xFF2A1515)
private val HamColor       = Color(0xFF43A047)
private val HamSurface     = Color(0xFF152A16)
private val Divider        = Color(0xFF2A2A2A)

data class SmsMessage(
    val id: String,
    val sender: String,
    val body: String,
    val timestamp: Long,
    var score: Float? = null,
    var flSent: Boolean = false
)

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var wordIndex: Map<String, Int>
    private val client = OkHttpClient()
    private val SERVER_URL = "http://20.189.113.13:8000"
    private val VOCAB_SIZE = 8000
    private val MAX_LEN = 100

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        loadModel()
        loadVocab()

        val clientId = Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID)

        setContent {
            SpamDetectorApp(
                onLoadSms       = { loadSmsMessages() },
                onClassify      = { text -> predict(text) },
                onSendToFL      = { text, cb -> sendFLUpdate(text, clientId, cb) }
            )
        }
    }

    private fun loadModel() {
        val afd     = assets.openFd("round_0.tflite")
        val fis     = FileInputStream(afd.fileDescriptor)
        val channel = fis.channel
        val buffer: MappedByteBuffer = channel.map(
            FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
        interpreter = Interpreter(buffer)
    }

    private fun loadVocab() {
        val json = assets.open("vocab.json").bufferedReader().readText()
        val obj  = JSONObject(json)
        val map  = mutableMapOf<String, Int>()
        obj.keys().forEach { key -> map[key] = obj.getInt(key) }
        wordIndex = map
    }

    private fun preprocess(text: String): String =
        text.lowercase()
            .replace(Regex("http\\S+|www\\S+"), "")
            .replace(Regex("\\d+"), "")
            .replace(Regex("[^a-z\\s]"), "")
            .replace(Regex("\\s+"), " ")
            .trim()

    private fun tokenize(text: String): FloatArray {
        val cleaned = preprocess(text)
        val tokens  = cleaned.split(" ").map { w ->
            val idx = wordIndex[w] ?: 1
            if (idx < VOCAB_SIZE) idx.toFloat() else 1f
        }
        return FloatArray(MAX_LEN) { i -> if (i < tokens.size) tokens[i] else 0f }
    }

    fun predict(text: String): Float {
        val input  = Array(1) { tokenize(text) }
        val output = Array(1) { FloatArray(1) }
        interpreter.run(input, output)
        return output[0][0]
    }

    fun loadSmsMessages(): List<SmsMessage> {
        val messages = mutableListOf<SmsMessage>()
        try {
            val uri    = Uri.parse("content://sms/inbox")
            val cursor = contentResolver.query(
                uri,
                arrayOf("_id", "address", "body", "date"),
                null, null, "date DESC LIMIT 50"
            ) ?: return messages

            cursor.use {
                val idIdx   = it.getColumnIndex("_id")
                val addrIdx = it.getColumnIndex("address")
                val bodyIdx = it.getColumnIndex("body")
                val dateIdx = it.getColumnIndex("date")
                while (it.moveToNext()) {
                    messages.add(SmsMessage(
                        id        = it.getString(idIdx) ?: "",
                        sender    = it.getString(addrIdx) ?: "Unknown",
                        body      = it.getString(bodyIdx) ?: "",
                        timestamp = it.getLong(dateIdx)
                    ))
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
        return messages
    }

    fun sendFLUpdate(message: String, clientId: String, onResult: (String) -> Unit) {
        val tokens     = tokenize(message)
        val tokenArray = JSONArray().apply { tokens.forEach { put(it.toDouble()) } }
        val json = JSONObject().apply {
            put("tokens",      tokenArray)
            put("label",       1)
            put("num_samples", 1)
            put("client_id",   clientId)
        }
        val body    = json.toString().toRequestBody("application/json".toMediaType())
        val request = Request.Builder().url("$SERVER_URL/upload_weights").post(body).build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: java.io.IOException) {
                onResult("Failed: ${e.message}")
            }
            override fun onResponse(call: Call, response: Response) {
                onResult(response.body?.string() ?: "")
            }
        })
    }
}

// ── Root composable ───────────────────────────────────
@Composable
fun SpamDetectorApp(
    onLoadSms:  () -> List<SmsMessage>,
    onClassify: (String) -> Float,
    onSendToFL: (String, (String) -> Unit) -> Unit
) {
    var hasPermission by remember { mutableStateOf(false) }
    var messages      by remember { mutableStateOf<List<SmsMessage>>(emptyList()) }
    var isLoading     by remember { mutableStateOf(false) }
    val scope         = rememberCoroutineScope()

    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission = granted
        if (granted) {
            scope.launch(Dispatchers.IO) {
                isLoading = true
                val loaded = onLoadSms()
                // Classify all messages
                val classified = loaded.map { msg ->
                    msg.copy(score = onClassify(msg.body))
                }
                withContext(Dispatchers.Main) {
                    messages  = classified
                    isLoading = false
                }
            }
        }
    }

    LaunchedEffect(Unit) {
        permLauncher.launch(Manifest.permission.READ_SMS)
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Background)
            .systemBarsPadding()
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            // Top bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(64.dp)
                    .padding(horizontal = 24.dp),
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    "FilterX",
                    style = TextStyle(
                        fontSize   = 20.sp,
                        fontWeight = FontWeight.W600,
                        color      = OnSurface,
                        letterSpacing = (-0.3).sp
                    )
                )
                Row(
                    verticalAlignment     = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(6.dp)
                            .clip(RoundedCornerShape(50))
                            .background(HamColor)
                    )
                    Text(
                        "FL Active",
                        style = TextStyle(
                            fontSize   = 12.sp,
                            color      = HamColor,
                            fontWeight = FontWeight.W500
                        )
                    )
                }
            }

            HorizontalDivider(color = Divider, thickness = 1.dp)

            when {
                !hasPermission -> PermissionDeniedScreen()
                isLoading      -> LoadingScreen()
                messages.isEmpty() -> EmptyScreen()
                else -> MessageList(
                    messages   = messages,
                    onMarkSpam = { msg ->
                        onSendToFL(msg.body) { }
                        messages = messages.map {
                            if (it.id == msg.id) it.copy(flSent = true) else it
                        }
                    },
                    onMarkHam  = { msg ->
                        messages = messages.map {
                            if (it.id == msg.id) it.copy(flSent = true) else it
                        }
                    }
                )
            }
        }
    }
}

// ── Message list ──────────────────────────────────────
@Composable
fun MessageList(
    messages:   List<SmsMessage>,
    onMarkSpam: (SmsMessage) -> Unit,
    onMarkHam:  (SmsMessage) -> Unit
) {
    val spamCount = messages.count { (it.score ?: 0f) >= 0.5f }

    Column(modifier = Modifier.fillMaxSize()) {
        // Summary bar
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 24.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                "${messages.size} messages",
                style = TextStyle(fontSize = 12.sp, color = OnSurfaceMuted)
            )
            Text(
                "$spamCount spam detected",
                style = TextStyle(fontSize = 12.sp, color = SpamColor, fontWeight = FontWeight.W500)
            )
        }

        LazyColumn(
            modifier            = Modifier.fillMaxSize(),
            contentPadding      = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(messages, key = { it.id }) { msg ->
                MessageCard(
                    message    = msg,
                    onMarkSpam = { onMarkSpam(msg) },
                    onMarkHam  = { onMarkHam(msg) }
                )
            }
        }
    }
}

// ── Message card ──────────────────────────────────────
@Composable
fun MessageCard(
    message:    SmsMessage,
    onMarkSpam: () -> Unit,
    onMarkHam:  () -> Unit
) {
    val score   = message.score ?: return
    val isSpam  = score >= 0.5f
    val cardBg  = if (isSpam) SpamSurface else Surface1
    val verdict = if (isSpam) "Spam" else "Ham"
    val vColor  = if (isSpam) SpamColor else HamColor
    val confPct = if (isSpam) score else (1f - score)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(cardBg)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        // Header row
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment     = Alignment.CenterVertically
        ) {
            Text(
                message.sender,
                style = TextStyle(
                    fontSize   = 13.sp,
                    fontWeight = FontWeight.W600,
                    color      = OnSurface
                )
            )
            Row(
                verticalAlignment     = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    "${"%.1f".format(confPct * 100)}%",
                    style = TextStyle(
                        fontSize   = 12.sp,
                        fontWeight = FontWeight.W600,
                        color      = vColor,
                        fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace
                    )
                )
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(4.dp))
                        .background(if (isSpam) Color(0xFF3D1515) else Color(0xFF152A16))
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                ) {
                    Text(
                        verdict,
                        style = TextStyle(
                            fontSize   = 10.sp,
                            fontWeight = FontWeight.W600,
                            color      = vColor,
                            letterSpacing = 0.5.sp
                        )
                    )
                }
            }
        }

        // Message body
        Text(
            message.body,
            style   = TextStyle(
                fontSize   = 14.sp,
                color      = OnSurfaceMuted,
                lineHeight = 20.sp
            ),
            maxLines = 3,
            overflow = TextOverflow.Ellipsis
        )

        // Confidence bar
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(3.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(Divider)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(confPct)
                    .height(3.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(vColor)
            )
        }

        // Action buttons
        if (!message.flSent) {
            Row(
                modifier              = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                // Not Spam button
                OutlinedButton(
                    onClick  = onMarkHam,
                    modifier = Modifier.weight(1f).height(36.dp),
                    shape    = RoundedCornerShape(6.dp),
                    border   = BorderStroke(1.dp, Divider),
                    colors   = ButtonDefaults.outlinedButtonColors(
                        contentColor = OnSurfaceMuted
                    )
                ) {
                    Text(
                        "Not Spam",
                        style = TextStyle(fontSize = 12.sp, fontWeight = FontWeight.W500)
                    )
                }

                // Spam button
                Button(
                    onClick  = onMarkSpam,
                    modifier = Modifier.weight(1f).height(36.dp),
                    shape    = RoundedCornerShape(6.dp),
                    colors   = ButtonDefaults.buttonColors(
                        containerColor = SpamColor
                    )
                ) {
                    Text(
                        "Spam — Train FL",
                        style = TextStyle(fontSize = 12.sp, fontWeight = FontWeight.W500)
                    )
                }
            }
        } else {
            Text(
                if (isSpam) "Contributed to FL" else "Marked as legitimate",
                style = TextStyle(
                    fontSize = 11.sp,
                    color    = if (isSpam) SpamColor else HamColor
                )
            )
        }
    }
}

// ── State screens ─────────────────────────────────────
@Composable
fun PermissionDeniedScreen() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp),
            modifier            = Modifier.padding(32.dp)
        ) {
            Text(
                "SMS permission required",
                style = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.W600, color = OnSurface)
            )
            Text(
                "Grant SMS access to classify your messages.",
                style     = TextStyle(fontSize = 14.sp, color = OnSurfaceMuted),
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )
        }
    }
}

@Composable
fun LoadingScreen() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            CircularProgressIndicator(color = Accent, modifier = Modifier.size(32.dp))
            Text(
                "Classifying messages...",
                style = TextStyle(fontSize = 14.sp, color = OnSurfaceMuted)
            )
        }
    }
}

@Composable
fun EmptyScreen() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Text(
            "No messages found",
            style = TextStyle(fontSize = 14.sp, color = OnSurfaceMuted)
        )
    }
}