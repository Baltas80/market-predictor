package com.baltas80.marketpredictor;

import android.app.Activity;
import android.os.Bundle;
import android.graphics.*;
import android.graphics.drawable.GradientDrawable;
import android.view.*;
import android.content.Context;
import java.util.Locale;

public class MainActivity extends Activity {
    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(new DashboardView(this));
    }

    static class DashboardView extends View {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        float d;
        int bg = Color.rgb(8,12,24), card = Color.rgb(17,24,39), text = Color.rgb(238,242,255), muted = Color.rgb(148,163,184), accent = Color.rgb(99,102,241), good = Color.rgb(34,197,94), warn = Color.rgb(245,158,11);
        DashboardView(Context c) { super(c); d = getResources().getDisplayMetrics().density; p.setTypeface(Typeface.create("sans", Typeface.NORMAL)); setLayerType(View.LAYER_TYPE_SOFTWARE, null); }
        float dp(float v){ return v*d; }
        void rect(Canvas c,float l,float t,float r,float b,float radius,int color){ p.setColor(color); p.setStyle(Paint.Style.FILL); c.drawRoundRect(dp(l),dp(t),dp(r),dp(b),dp(radius),dp(radius),p); }
        void txt(Canvas c,String s,float x,float y,float size,int color,boolean bold){ p.setColor(color);p.setTextSize(dp(size));p.setTypeface(Typeface.create("sans",bold?Typeface.BOLD:Typeface.NORMAL));p.setStyle(Paint.Style.FILL);c.drawText(s,dp(x),dp(y),p); }
        @Override protected void onDraw(Canvas c){
            c.drawColor(bg);
            float w=getWidth()/d;
            txt(c,"MARKET PREDICTOR",20,32,13,muted,true);
            txt(c,"Research Dashboard",20,62,26,text,true);
            txt(c,"RESEARCH ONLY  •  NO LIVE TRADING",20,86,11,warn,true);
            card(c,20,102,w-20,184,"DATA COVERAGE","2015 — 2025","Historical dataset",good);
            float gap=12, cw=(w-52)/2;
            card(c,20,196,20+cw,278,"A/B/C STATUS","READY","Technical / Macro / Events",good);
            card(c,32+cw,196,w-20,278,"LOCKBOX","PENDING","Waiting for final OOS",warn);
            txt(c,"Performance",20,314,19,text,true);
            card(c,20,328,w-20,470,"A/B/C FINANCIAL COMPARISON","","",accent);
            drawChart(c,38,366,w-38,442);
            txt(c,"A",42,458,11,muted,true); txt(c,"B",w/2-5,458,11,muted,true); txt(c,"C",w-58,458,11,muted,true);
            txt(c,"Lockbox metrics",20,506,19,text,true);
            metric(c,20,520, "ROC-AUC", "—", "Awaiting OOS");
            metric(c,160,520,"Sharpe","—","Costs included");
            metric(c,300,520,"Max DD","—","Risk metric");
            txt(c,"Provenance",20,640,19,text,true);
            rect(c,20,654,w-20,760,16,card);
            txt(c,"Dataset hash",36,682,11,muted,false); txt(c,"PENDING",w-94,682,11,warn,true);
            txt(c,"Code version",36,710,11,muted,false); txt(c,"LOCKBOX",w-94,710,11,good,true);
            txt(c,"Status",36,738,11,muted,false); txt(c,"READ-ONLY",w-94,738,11,accent,true);
            txt(c,"Prototype UI • Real results will replace placeholders after lockbox",20,790,10,muted,false);
        }
        void card(Canvas c,float l,float t,float r,float b,String title,String value,String sub,int state){
            rect(c,l,t,r,b,16,card); txt(c,title,l+16,t+25,10,muted,true); txt(c,value,l+16,t+55,22,text,true); txt(c,sub,l+16,t+74,10,muted,false); rect(c,r-48,t+16,r-16,t+48,16,state); }
        void metric(Canvas c,float x,float y,String title,String value,String sub){ rect(c,x,y,x+124,y+100,14,card);txt(c,title,x+12,y+23,10,muted,true);txt(c,value,x+12,y+55,20,text,true);txt(c,sub,x+12,y+78,9,muted,false); }
        void drawChart(Canvas c,float l,float t,float r,float b){
            p.setColor(Color.rgb(51,65,85));p.setStrokeWidth(dp(1));p.setStyle(Paint.Style.STROKE);c.drawRoundRect(dp(l),dp(t),dp(r),dp(b),dp(10),dp(10),p);
            Path path=new Path(); float[] vals={.24f,.31f,.28f,.42f,.38f,.52f,.49f,.63f,.58f,.71f,.68f,.79f};
            p.setColor(accent);p.setStrokeWidth(dp(3));p.setStyle(Paint.Style.STROKE);
            for(int i=0;i<vals.length;i++){float x=dp(l+12+(r-l-24)*i/(vals.length-1));float y=dp(b-12-(b-t-24)*vals[i]);if(i==0)path.moveTo(x,y);else path.lineTo(x,y);}c.drawPath(path,p);
        }
    }
}
