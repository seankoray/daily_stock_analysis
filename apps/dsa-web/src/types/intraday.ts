export interface IntradaySignal {
  code: string;
  status: 'watch' | 'triggered' | string;
  direction: 'watch' | 'sell_then_buy' | 'buy_then_sell' | string;
  suggestedQuantity: number;
  signalKey?: string | null;
  reason?: string | null;
  reasons: string[];
  candidateBuyRange?: Array<number | null> | null;
  candidateSellRange?: Array<number | null> | null;
  triggerPrice?: number | null;
  invalidationPrice?: number | null;
  dataFresh: boolean;
  dataAgeSeconds?: number | null;
  signalChanged: boolean;
  advisoryOnly: boolean;
  portfolioContext: Record<string, unknown>;
  technical: Record<string, unknown>;
}
