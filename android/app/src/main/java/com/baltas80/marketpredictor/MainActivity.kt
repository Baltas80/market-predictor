package com.baltas80.marketpredictor

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.view.Window

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = Color.rgb(7, 17, 29)
        window.navigationBarColor = Color.rgb(7, 17, 29)
        setContentView(MarketPredictorView(this))
    }
}
