package com.baltas80.marketpredictor

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.view.MotionEvent
import android.view.View
import kotlin.math.max

class MarketPredictorView(context: Context) : View(context) {
    private val bg = 0xFF07111D.toInt()
    private val panel = 0xFF0D1A28.toInt()
    private val panel2 = 0xFF112337.toInt()
    private val text = 0xFFF3F7FB.toInt()
    private val muted = 0xFF8EA1B5.toInt()
    private val cyan = 0xFF22B8F0.toInt()
    private val green = 0xFF25D98B.toInt()
    private val red = 0xFFFF5E68.toInt()
    private val orange = 0xFFFFB648.toInt()
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var screen = 0
    private var nav = 0
    private val names = arrayOf("Inicio", "Mercados", "Agentes", "Cartera", "Más")

    init { setBackgroundColor(bg); isClickable = true }

    private fun d(v: Float) = v * resources.displayMetrics.density
    private fun txt(canvas: Canvas, s: String, x: Float, y: Float, size: Float, color: Int = text, bold: Boolean = false) {
        paint.style = Paint.Style.FILL; paint.color = color; paint.textSize = d(size)
        paint.typeface = if (bold) android.graphics.Typeface.DEFAULT_BOLD else android.graphics.Typeface.DEFAULT
        canvas.drawText(s, d(x), d(y), paint)
    }
    private fun round(canvas: Canvas, l: Float, t: Float, r: Float, b: Float, radius: Float, color: Int) {
        paint.style = Paint.Style.FILL; paint.color = color
        canvas.drawRoundRect(RectF(d(l), d(t), d(r), d(b)), d(radius), d(radius), paint)
    }
    private fun line(canvas: Canvas, x1: Float, y1: Float, x2: Float, y2: Float, color: Int, width: Float = 1f) {
        paint.color = color; paint.strokeWidth = d(width); paint.style = Paint.Style.STROKE
        canvas.drawLine(d(x1), d(y1), d(x2), d(y2), paint)
    }

    override fun onDraw(c: Canvas) {
        super.onDraw(c)
        c.drawColor(bg)
        when (screen) { 0 -> dashboard(c); 1 -> markets(c); 2 -> agents(c); 3 -> portfolio(c); else -> more(c) }
        bottomNav(c)
    }

    private fun header(c: Canvas, title: String) {
        txt(c, "▰", 18f, 31f, 20f, cyan, true)
        txt(c, "Market Predictor", 44f, 30f, 16f, text, true)
        txt(c, title, 18f, 61f, 26f, text, true)
        round(c, 250f, 14f, 352f, 40f, 13f, panel2)
        txt(c, "●  PAPER", 265f, 32f, 11f, green, true)
    }

    private fun dashboard(c: Canvas) {
        header(c, "Hola, inversor")
        txt(c, "La inteligencia de los mercados en tu mano.", 18f, 82f, 12f, muted)
        round(c, 18f, 94f, 352f, 126f, 12f, panel)
        txt(c, "CARTERA", 32f, 114f, 10f, muted, true)
        txt(c, "25.420,18 €", 32f, 148f, 27f, text, true)
        txt(c, "+2,36%   (+586,42 €)", 32f, 169f, 13f, green, true)
        chart(c, 30f, 181f, 338f, 215f)
        metric(c, 18f, 225f, "EXPOSICIÓN", "68%", cyan)
        metric(c, 136f, 225f, "DRAWDOWN", "4,2%", orange)
        metric(c, 254f, 225f, "RÉGIMEN", "ALCISTA", green)
        txt(c, "Señales principales", 18f, 298f, 17f, text, true)
        signal(c, 18f, 310f, "S&P 500", "ALCISTA", "68%", green)
        signal(c, 18f, 371f, "NASDAQ", "ALCISTA", "64%", green)
        signal(c, 18f, 432f, "ORO", "NEUTRAL", "51%", orange)
        round(c, 18f, 497f, 352f, 553f, 12f, panel)
        txt(c, "GUARDIAN · RISK GATE", 32f, 520f, 11f, muted, true)
        txt(c, "● Operativo", 32f, 541f, 14f, green, true)
    }

    private fun markets(c: Canvas) {
        header(c, "Mercados")
        round(c, 18f, 72f, 352f, 108f, 12f, panel)
        txt(c, "S&P 500", 32f, 95f, 15f, text, true); txt(c, "5.472,18", 32f, 126f, 22f, text, true); txt(c, "+0,82%", 270f, 126f, 13f, green, true)
        chart(c, 32f, 137f, 338f, 170f)
        txt(c, "Índices", 18f, 206f, 12f, cyan, true)
        val(c, 18f, 224f, "S&P 500", "5.472,18", "+0,82%", green)
        val(c, 18f, 270f, "NASDAQ", "17.280,45", "+1,14%", green)
        val(c, 18f, 316f, "DAX", "18.647,21", "+0,73%", green)
        val(c, 18f, 362f, "IBEX 35", "11.231,90", "+0,66%", green)
        val(c, 18f, 408f, "Nikkei 225", "32.718,56", "−0,32%", red)
        val(c, 18f, 454f, "EUR/USD", "1,0762", "−0,21%", red)
        val(c, 18f, 500f, "Bitcoin", "56.248,00", "+2,14%", green)
    }

    private fun agents(c: Canvas) {
        header(c, "Agentes IA")
        txt(c, "Especialistas que analizan. GUARDIAN supervisa.", 18f, 82f, 12f, muted)
        agent(c, 18f, 105f, "ORÁCULO", "Macro y régimen de mercado", "Activo", cyan)
        agent(c, 18f, 185f, "ATLAS", "Tendencias y análisis técnico", "Activo", green)
        agent(c, 18f, 265f, "NEWS AI", "Noticias y eventos", "Paper", orange)
        agent(c, 18f, 345f, "GUARDIAN", "Gestor de riesgo independiente", "Siempre activo", green)
        round(c, 18f, 450f, 352f, 535f, 12f, panel)
        txt(c, "DECISIÓN CONJUNTA", 32f, 474f, 10f, muted, true)
        txt(c, "ALCISTA", 32f, 505f, 22f, green, true)
        txt(c, "Confianza calibrada · 68%", 145f, 504f, 12f, text)
        txt(c, "Última actualización · PIT 09:41 UTC", 32f, 525f, 10f, muted)
    }

    private fun portfolio(c: Canvas) {
        header(c, "Mi cartera")
        txt(c, "Valor total", 18f, 87f, 11f, muted); txt(c, "25.420,18 €", 18f, 119f, 26f, text, true); txt(c, "+2,36%", 278f, 119f, 13f, green, true)
        round(c, 18f, 137f, 352f, 206f, 12f, panel)
        txt(c, "DISTRIBUCIÓN", 32f, 160f, 10f, muted, true)
        txt(c, "Acciones", 32f, 185f, 13f, text); txt(c, "52%", 310f, 185f, 13f, text, true)
        txt(c, "ETF", 32f, 208f, 13f, text); txt(c, "24%", 310f, 208f, 13f, text, true)
        txt(c, "Efectivo", 32f, 231f, 13f, text); txt(c, "12%", 310f, 231f, 13f, text, true)
        txt(c, "Otros", 32f, 254f, 13f, text); txt(c, "12%", 310f, 254f, 13f, text, true)
        txt(c, "Mis posiciones", 18f, 292f, 17f, text, true)
        position(c, 18f, 310f, "AAPL", "4.210,32 €", "+3,12%", green)
        position(c, 18f, 358f, "MSFT", "3.980,11 €", "+1,26%", green)
        position(c, 18f, 406f, "SPY", "5.632,44 €", "+0,82%", green)
        position(c, 18f, 454f, "GOLD", "2.140,00 €", "−0,45%", red)
    }

    private fun more(c: Canvas) {
        header(c, "Más")
        val rows = arrayOf("Estrategias", "Backtesting", "Noticias y eventos", "Ejecución", "Riesgo y seguridad", "Ajustes")
        rows.forEachIndexed { i, s ->
            val y = 85f + i * 58f
            round(c, 18f, y, 352f, y + 46f, 11f, panel)
            txt(c, s, 34f, y + 29f, 14f, text, true); txt(c, "›", 326f, y + 29f, 20f, muted)
        }
    }

    private fun bottomNav(c: Canvas) {
        val y = height / resources.displayMetrics.density - 64f
        round(c, 8f, y, 362f, y + 58f, 16f, panel)
        for (i in 0..4) {
            val x = 12f + i * 70f
            val color = if (i == nav) cyan else muted
            txt(c, if (i == nav) "●" else "○", x + 24f, y + 22f, 13f, color, true)
            txt(c, names[i], x + 8f, y + 42f, 9f, color, i == nav)
        }
    }

    private fun metric(c: Canvas, x: Float, y: Float, label: String, value: String, color: Int) { round(c, x, y, x+106f, y+52f, 10f, panel); txt(c, label, x+10f, y+20f, 8f, muted, true); txt(c, value, x+10f, y+40f, 13f, color, true) }
    private fun signal(c: Canvas, x: Float, y: Float, asset: String, dir: String, conf: String, color: Int) { round(c,x,y,x+334f,y+51f,10f,panel); txt(c,asset,x+14f,y+21f,13f,text,true); txt(c,dir,x+135f,y+21f,11f,color,true); txt(c,conf,x+286f,y+21f,11f,color,true); txt(c,"Modelo v1.2 · PIT 09:41",x+14f,y+39f,9f,muted) }
    private fun val(c: Canvas,x:Float,y:Float,a:String,p:String,v:String,color:Int){txt(c,a,x,y+15f,13f,text,true);txt(c,p,145f,y+15f,13f,text);txt(c,v,295f,y+15f,11f,color,true);line(c,18f,y+27f,352f,y+27f,panel2)}
    private fun agent(c: Canvas,x:Float,y:Float,name:String,desc:String,status:String,color:Int){round(c,x,y,x+334f,y+66f,12f,panel);round(c,x+12f,y+12f,x+45f,y+45f,17f,panel2);txt(c,"✦",x+22f,y+34f,15f,color,true);txt(c,name,x+58f,y+23f,13f,text,true);txt(c,desc,x+58f,y+42f,10f,muted);txt(c,status,x+258f,y+23f,9f,color,true)}
    private fun position(c:Canvas,x:Float,y:Float,s:String,p:String,v:String,color:Int){round(c,x,y,x+334f,y+39f,9f,panel);txt(c,s,x+13f,y+25f,13f,text,true);txt(c,p,130f,y+25f,12f,text);txt(c,v,286f,y+25f,10f,color,true)}
    private fun chart(c: Canvas,l:Float,t:Float,r:Float,b:Float){
        paint.style=Paint.Style.STROKE; paint.strokeWidth=d(2f); paint.color=green
        val path=Path(); val w=r-l; val h=b-t
        path.moveTo(d(l),d(b-10)); path.lineTo(d(l+w*.10f),d(b-h*.25f)); path.lineTo(d(l+w*.18f),d(b-h*.18f)); path.lineTo(d(l+w*.30f),d(b-h*.45f)); path.lineTo(d(l+w*.40f),d(b-h*.36f)); path.lineTo(d(l+w*.53f),d(b-h*.62f)); path.lineTo(d(l+w*.65f),d(b-h*.55f)); path.lineTo(d(l+w*.78f),d(b-h*.78f)); path.lineTo(d(r),d(b-h*.90f)); canvasPath(c,path)
        line(c,l,t+h*.25f,r,t+h*.25f,panel2); line(c,l,t+h*.55f,r,t+h*.55f,panel2); line(c,l,t+h*.85f,r,t+h*.85f,panel2)
    }
    private fun canvasPath(c:Canvas,p:Path){c.drawPath(p,paint)}

    override fun onTouchEvent(e: MotionEvent): Boolean {
        if (e.action != MotionEvent.ACTION_UP) return true
        val x=e.x/resources.displayMetrics.density; val y=e.y/resources.displayMetrics.density
        val navY=height/resources.displayMetrics.density-72f
        if(y>=navY){ nav=(x/72f).toInt().coerceIn(0,4); screen=nav; invalidate(); return true }
        if(screen==0 && y in 300f..490f){ screen=1; nav=1; invalidate() }
        return true
    }
}
