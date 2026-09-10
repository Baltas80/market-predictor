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
        scroll.setBackgroundColor(Color.rgb(6,10,20));
        scroll.setVerticalScrollBarEnabled(false);
        scroll.addView(new DashboardView(this));
        setContentView(scroll);
    }

    static class DashboardView extends View {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        float d;
        int bg = Color.rgb(6,10,20), card = Color.rgb(15,23,39), card2 = Color.rgb(18,27,45);
        int text = Color.rgb(241,245,255), muted = Color.rgb(148,163,184), accent = Color.rgb(99,102,241);
        int good = Color.rgb(34,197,94), warn = Color.rgb(245,158,11), border = Color.rgb(38,52,74);
        DashboardView(Context c) { super(c); d = getResources().getDisplayMetrics().density; setBackgroundColor(bg); setMinimumHeight((int)dp(900)); }
        float dp(float v){ return v*d; }
        void rect(Canvas c,float l,float t,float r,float b,float radius,int color){ p.setColor(color); p.setStyle(Paint.Style.FILL); c.drawRoundRect(dp(l),dp(t),dp(r),dp(b),dp(radius),dp(radius),p); }
        void outline(Canvas c,float l,float t,float r,float b,float radius,int color){ p.setColor(color); p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(dp(1)); c.drawRoundRect(dp(l),dp(t),dp(r),dp(b),dp(radius),dp(radius),p); }
        void txt(Canvas c,String s,float x,float y,float size,int color,boolean bold){ p.setColor(color); p.setTextSize(dp(size)); p.setTypeface(Typeface.create("sans",bold?Typeface.BOLD:Typeface.NORMAL)); p.setStyle(Paint.Style.FILL); c.drawText(s,dp(x),dp(y),p); }
        void dot(Canvas c,float x,float y,int color){ p.setColor(color); p.setStyle(Paint.Style.FILL); c.drawCircle(dp(x),dp(y),dp(6),p); p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(dp(1)); p.setColor(Color.argb(90,255,255,255)); c.drawCircle(dp(x),dp(y),dp(7),p); }
        @Override protected void onDraw(Canvas c){
            super.onDraw(c); c.drawColor(bg);
            float w=getWidth()/d;
            float margin=Math.max(18, Math.min(24, w*0.05f));
            txt(c,"MARKET PREDICTOR",margin,32,13,muted,true);
            txt(c,"Research Dashboard",margin,62,26,text,true);
            txt(c,"RESEARCH ONLY  •  NO LIVE TRADING",margin,86,11,warn,true);

            premiumCard(c,margin,102,w-margin,184);
            txt(c,"DATA COVERAGE",margin+16,127,10,muted,true);
            txt(c,"2015 — 2025",margin+16,157,22,text,true);
            txt(c,"Historical dataset • ",margin+16,177,10,muted,false);
            txt(c,"AVAILABLE",margin+110,177,10,good,true);
            dot(c,w-margin-30,143,good);

            float gap=12, cw=(w-2*margin-gap)/2;
            premiumCard(c,margin,196,margin+cw,278);
            txt(c,"A/B/C STATUS",margin+16,221,10,muted,true);
            float configuredSize = cw < 145 ? 18 : 21;
            txt(c,"CONFIGURED",margin+16,252,configuredSize,text,true);
            txt(c,"Technical / Macro / Events",margin+16,272,10,muted,false);
            dot(c,margin+cw-30,228,good);

            float rx=margin+cw+gap;
            premiumCard(c,rx,196,w-margin,278);
            txt(c,"LOCKBOX",rx+16,221,10,muted,true);
            txt(c,"PENDING",rx+16,252,21,text,true);
            txt(c,"Waiting for final OOS",rx+16,272,10,muted,false);
            dot(c,w-margin-30,228,warn);

            txt(c,"Performance",margin,314,19,text,true);
            premiumCard(c,margin,328,w-margin,474);
            txt(c,"A/B/C FINANCIAL COMPARISON",margin+16,353,10,muted,true);
            p.setTextAlign(Paint.Align.RIGHT); txt(c,"PREVIEW • MOCK DATA",w-margin-16,353,9,warn,true); p.setTextAlign(Paint.Align.LEFT);
            drawChart(c,margin+14,366,w-margin-14,440);
            txt(c,"A",margin+8,462,11,muted,true); txt(c,"B",w/2-5,462,11,muted,true); txt(c,"C",w-margin-18,462,11,muted,true);

            txt(c,"Lockbox metrics",margin,510,19,text,true);
            float mw=(w-2*margin-2*gap)/3;
            metric(c,margin,524,mw,"ROC-AUC","—","Awaiting OOS");
            metric(c,margin+mw+gap,524,mw,"Sharpe","—","Costs included");
            metric(c,margin+2*(mw+gap),524,mw,"Max DD","—","Risk metric");

            txt(c,"Provenance",margin,656,19,text,true);
            premiumCard(c,margin,670,w-margin,816);
            provenance(c,"Dataset hash","PENDING",696,w,warn,margin);
            provenance(c,"Code version","LOCKBOX",724,w,good,margin);
            provenance(c,"Protocol","PENDING",752,w,warn,margin);
            provenance(c,"Mode","READ-ONLY",780,w,accent,margin);
            txt(c,"Prototype UI • real results replace placeholders after lockbox",margin+16,804,9,muted,false);
        }
        void premiumCard(Canvas c,float l,float t,float r,float b){
            rect(c,l,t,r,b,17,card);
            outline(c,l,t,r,b,17,border);
        }
        void metric(Canvas c,float x,float y,float width,String title,String value,String sub){
            rect(c,x,y,x+width,y+100,14,card2); outline(c,x,y,x+width,y+100,14,border);
            txt(c,title,x+12,y+23,10,muted,true); txt(c,value,x+12,y+55,20,text,true); txt(c,sub,x+12,y+78,9,muted,false);
        }
        void provenance(Canvas c,String label,String value,float y,float w,int color,float margin){
            txt(c,label,margin+16,y,11,muted,false); p.setTextAlign(Paint.Align.RIGHT); txt(c,value,w-margin-16,y,11,color,true); p.setTextAlign(Paint.Align.LEFT);
        }
        void drawChart(Canvas c,float l,float t,float r,float b){
            p.setColor(border);p.setStrokeWidth(dp(1));p.setStyle(Paint.Style.STROKE);c.drawRoundRect(dp(l),dp(t),dp(r),dp(b),dp(10),dp(10),p);
            Path path=new Path(); float[] vals={.24f,.31f,.28f,.42f,.38f,.52f,.49f,.63f,.58f,.71f,.68f,.79f};
            p.setColor(accent);p.setStrokeWidth(dp(3));p.setStyle(Paint.Style.STROKE);
            for(int i=0;i<vals.length;i++){float x=dp(l+12+(r-l-24)*i/(vals.length-1));float y=dp(b-12-(b-t-24)*vals[i]);if(i==0)path.moveTo(x,y);else path.lineTo(x,y);}c.drawPath(path,p);
        }
    }
}
