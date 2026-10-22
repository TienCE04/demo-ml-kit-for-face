package com.leansoft.draw.mlkitforface.attendance

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.nio.ByteBuffer
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
import androidx.core.content.edit

/** Small encrypted SharedPreferences adapter for local demo data. */
class SecurePreferenceStore(context: Context, name: String) {
    private val preferences = context.getSharedPreferences(name, Context.MODE_PRIVATE)
    private val alias = "${name}_aes_gcm_v1"

    fun getString(key: String, defaultValue: String? = null): String? {
        val encoded = preferences.getString(key, null) ?: return defaultValue
        // Preserve data created by the earlier plain SharedPreferences demo;
        // the next write transparently upgrades that value to AES-GCM.
        return runCatching { decrypt(encoded) }.getOrElse { encoded }
    }

    fun putString(key: String, value: String) {
        preferences.edit { putString(key, encrypt(value)) }
    }

    fun getFloat(key: String, defaultValue: Float): Float =
        getString(key)?.toFloatOrNull() ?: defaultValue

    fun putFloat(key: String, value: Float) = putString(key, value.toString())

    private fun encrypt(value: String): String {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, key())
        val encrypted = cipher.doFinal(value.toByteArray(Charsets.UTF_8))
        val payload = ByteBuffer.allocate(4 + cipher.iv.size + encrypted.size)
            .putInt(cipher.iv.size)
            .put(cipher.iv)
            .put(encrypted)
            .array()
        return Base64.encodeToString(payload, Base64.NO_WRAP)
    }

    private fun decrypt(encoded: String): String {
        val payload = Base64.decode(encoded, Base64.NO_WRAP)
        val buffer = ByteBuffer.wrap(payload)
        val ivSize = buffer.int
        require(ivSize in 12..16)
        val iv = ByteArray(ivSize).also(buffer::get)
        val encrypted = ByteArray(buffer.remaining()).also(buffer::get)
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(TAG_BITS, iv))
        return cipher.doFinal(encrypted).toString(Charsets.UTF_8)
    }

    private fun key(): SecretKey {
        val store = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        (store.getKey(alias, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(
                alias,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            ).setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .build()
        )
        return generator.generateKey()
    }

    companion object {
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val TAG_BITS = 128
        private const val ANDROID_KEYSTORE = "AndroidKeyStore"
    }
}
