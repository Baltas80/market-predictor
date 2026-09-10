package com.baltas80.marketpredictor;

import android.app.Activity;
import android.os.Bundle;
import android.graphics.*;
import android.view.*;
import android.widget.ScrollView;
import android.content.Context;

public class MainActivity extends Activity {
    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(Color.rgb(8,12,24));
        scroll.addView(new DashboardView(this));
        setContentView(scroll);
    }

    static class DashboardView extends View {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        float d;
        int bg = Color.rgb(8,12,24), card = Color.rgb(17,24,39), text = Color.rgb(238,242,255), muted = Color.rgb(148,163,184), accent = Color.rgb(99,102,241), good = Color.rgb(34,197,94), warn = Color.rgb(245,158,11);
        DashboardView(Context c) { super(c); d = getResources().getDisplayMetrics().density; p.setTypeface(Typeface.create("sans", Typeface.NORMAL)); setBackgroundColor(bg); setMinimumHeight((int)dp(900)); }
        float dp(float v){ return v*d; }
        void rect(Canvas c,float l,float t,float r,float b,float radius,int color){ p.setColor(color); p.setStyle(Paint.Style.FILL); c.drawRoundRect(dp(l),dp(t),dp(r),dp(b),dp(radius),dp(radius),p); }
        void txt(Canvas c,String s,float x,float y,float size,int color,boolean bold){ p.setColor(color);p.setTextSize(dp(size));p.setTypeface(Typeface.create("sans",bold?Typeface.BOLD:Typeface.NORMAL));p.setStyle(Paint.Style.FILL);c.drawText(s,dp(x),dp(y),p); }
        @Override protected void onDraw(Canvas c){
            super.onDraw(c);
            c.drawColor(bg);
            float w=getWidth()/d;
            float margin=Math.max(18, Math.min(24, w*0.05f));
            txt(c,"MARKET PREDICTOR",margin,32,13,muted,true);
            txt(c,"Research Dashboard",margin,62,26,text,true);
            txt(c,"RESEARCH ONLY  •  NO LIVE TRADING",margin,86,11,warn,true);

            card(c,margin,102,w-margin,184,"DATA COVERAGE","2015 — 2025","Historical dataset • AVAILABLE",good);
            float gap=12, cw=(w-2*margin-gap)/2;
            card(c,margin,196,margin+cw,278,"A/B/C STATUS","CONFIGURED","Technical / Macro / Events",good);
            card(c,margin+cw+gap,196,w-margin,278,"LOCKBOX","PENDING","Waiting for final OOS",warn);

            txt(c,"Performance",margin,314,19,text,true);
            rect(c,margin,328,w-margin,474,16,card);
            txt(c,"A/B/C FINANCIAL COMPARISON",margin+16,353,10,muted,true);
            p.setTextAlign(Paint.Align.RIGHT); txt(c,"PREVIEW • MOCK DATA",w-margin-16,353,9,warn,true); p.setTextAlign(Paint.Align.LEFT);
            drawChart(c,margin+14,366,w-margin-14,440);
            txt(c,"A",margin+8,462,11,muted,true); txt(c,"B",w/2-5,462,11,muted,true); txt(c,"C",w-margin-18,462,11,muted,true);

            txt(c,"Lockbox metrics",margin,510,19,text,true);
            float mw=Math.max(86,(w-2*margin-2*gap)/3);
            metric(c,margin,524,mw,"ROC-AUC","—","Awaiting OOS");
            metric(c,margin+mw+gap,524,mw,"Sharpe","—","Costs included");
            metric(c,margin+2*(mw+gap),524,mw,"Max DD","—","Risk metric");

            txt(c,"Provenance",margin,656,19,text,true);
            rect(c,margin,670,w-margin,816,16,card);
            provenance(c,"Dataset hash","PENDING",696,w,warn,margin);
            provenance(c,"Code version","LOCKBOX",724,w,good,margin);
            provenance(c,"Protocol","PENDING",752,w,warn,margin);
            provenance(c,"Mode","READ-ONLY",780,w,accent,margin);
            txt(c,"Prototype UI • real results replace placeholders after lockbox",margin,804,9,muted,false);
        }
        void card(Canvas c,float l,float t,float r,float b,String title,String value,String sub,int state){
            rect(c,l,t,r,b,16,card); txt(c,title,l+16,t+25,10,muted,true); txt(c,value,l+16,t+55,22,text,true); txt(c,sub,l+16,t+74,10,muted,false); rect(c,r-48,t+16,r-16,t+48,16,state);
        }
        void metric(Canvas c,float x,float y,float width,String title,String value,String sub){
            rect(c,x,y,x+width,y+100,14,card); txt(c,title,x+12,y+23,10,muted,true); txt(c,value,x+12,y+55,20,text,true); txt(c,sub,x+12,y+78,9,muted,false);
        }
        void provenance(Canvas c,String label,String value,float y,float w,int color,float margin){
            txt(c,label,margin+16,y,11,muted,false); p.setTextAlign(Paint.Align.RIGHT); txt(c,value,w-margin-16,y,11,color,true); p.setTextAlign(Paint.Align.LEFT);
        }
        void drawChart(Canvas c,float l,float t,float r,float b){
            p.setColor(Color.rgb(51,65,85));p.setStrokeWidth(dp(1));p.setStyle(Paint.Style.STROKE);c.drawRoundRect(dp(l),dp(t),dp(r),dp(b),dp(10),dp(10),p);
            Path path=new Path(); float[] vals={.24f,.31f,.28f,.42f,.38f,.52f,.49f,.63f,.58f,.71f,.68f,.79f};
            p.setColor(accent);p.setStrokeWidth(dp(3));p.setStyle(Paint.Style.STROKE);
            for(int i=0;i<vals.length;i++){float x=dp(l+12+(r-l-24)*i/(vals.length-1));float y=dp(b-12-(b-t-24)*vals[i]);if(i==0)path.moveTo(x,y);else path.lineTo(x,y);}c.drawPath(path,p);
        }
    }
}
