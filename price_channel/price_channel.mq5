//+------------------------------------------------------------------+
//|                                                price_channel.mq5 |
//|                        Copyright 2019, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include  <0.mqh>
op o ;

double n = 0.005;

double myma_in_low;
double values_in_low[];
double myma_in_high;
double values_in_high[];
double top_in_price;
double bot_in_price;

int n_in = 100;



//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
//--- create timer
   EventSetTimer(60);

//---
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
//--- destroy timer
   EventKillTimer();

}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
//---
   myma_in_low = iMA(NULL,PERIOD_CURRENT,1,0,MODE_SMA,PRICE_LOW) ;
   CopyBuffer(myma_in_low,0,1,n_in,values_in_low);
   myma_in_high = iMA(NULL,PERIOD_CURRENT,1,0,MODE_SMA,PRICE_HIGH) ;
   CopyBuffer(myma_in_high,0,1,n_in,values_in_high);

   ArraySetAsSeries(values_in_high,true);
   top_in_price = values_in_high[ArrayMaximum(values_in_high,0,WHOLE_ARRAY)];
   ArraySetAsSeries(values_in_low,true);
   bot_in_price = values_in_low[ArrayMinimum(values_in_low,0,WHOLE_ARRAY)];

   //Comment("top_in_price:",top_in_price,"\n","bot_in_price:",bot_in_price,"\n");


   if(PositionsTotal()==0)
   {
      if(SymbolInfoDouble(Symbol(),SYMBOL_BID) > top_in_price)
      {
         o.t_buy(0.01,888);
         o.sltp(SymbolInfoDouble(Symbol(),SYMBOL_BID)-n,SymbolInfoDouble(Symbol(),SYMBOL_BID)+n,888);
      }

      if(SymbolInfoDouble(Symbol(),SYMBOL_BID) < bot_in_price)
      {
         o.t_sell(0.01,666);
         o.sltp(SymbolInfoDouble(Symbol(),SYMBOL_BID)+n,SymbolInfoDouble(Symbol(),SYMBOL_BID)-n,666);

      }
   }






}
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
//---

}
//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade()
{
//---

}
//+------------------------------------------------------------------+
//| TradeTransaction function                                        |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
//---

}
//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
{
//---
   double ret=0.0;
//---

//---
   return(ret);
}
//+------------------------------------------------------------------+
//| TesterInit function                                              |
//+------------------------------------------------------------------+
void OnTesterInit()
{
//---

}
//+------------------------------------------------------------------+
//| TesterPass function                                              |
//+------------------------------------------------------------------+
void OnTesterPass()
{
//---

}
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
{
//---

}
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
{
//---

}
//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
{
//---

}
//+------------------------------------------------------------------+
