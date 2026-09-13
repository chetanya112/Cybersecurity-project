import axios from 'axios';
import type { DatasetInfo, PredictionResponse, CompatibilityReport } from './types';

const API_BASE_URL = 'http://localhost:8000/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getDatasetInfo = async (): Promise<DatasetInfo> => {
  const { data } = await apiClient.get<DatasetInfo>('/dataset/info');
  return data;
};

export const getSequencePrediction = async (index: number): Promise<PredictionResponse> => {
  const { data } = await apiClient.get<PredictionResponse>(`/dataset/sequence/${index}`);
  return data;
};

export const uploadCSV = async (file: File, onProgress?: (progressEvent: any) => void): Promise<CompatibilityReport> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const { data } = await apiClient.post<CompatibilityReport>('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: onProgress,
  });
  return data;
};

export const analyzeTraffic = async (uploadId: string): Promise<PredictionResponse> => {
  const { data } = await apiClient.post<PredictionResponse>(`/analyze/${uploadId}`);
  return data;
};
