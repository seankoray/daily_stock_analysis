import apiClient from './index';
import { toCamelCase } from './utils';
import type { IntradaySignal } from '../types/intraday';

export const intradayApi = {
  async getSignal(stockCode: string, accountId?: number): Promise<IntradaySignal> {
    const response = await apiClient.get<Record<string, unknown>>(`/api/v1/intraday/${stockCode}/signal`, {
      params: accountId == null ? undefined : { account_id: accountId },
    });
    return toCamelCase<IntradaySignal>(response.data);
  },

  async refresh(stockCode: string, accountId?: number): Promise<IntradaySignal> {
    const response = await apiClient.post<Record<string, unknown>>(
      `/api/v1/intraday/${stockCode}/refresh`,
      undefined,
      { params: accountId == null ? undefined : { account_id: accountId } },
    );
    return toCamelCase<IntradaySignal>(response.data);
  },
};
