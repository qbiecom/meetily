import { invoke } from '@tauri-apps/api/core';

export type QuantizationType = 'FP32' | 'FP16' | 'Int8';

export type ModelStatus =
  | 'Available'
  | 'Missing'
  | { Available: null }
  | { Missing: null }
  | { Downloading: { progress: number } }
  | { Error: string }
  | { Corrupted: { file_size: number; expected_min_size: number } };

export interface SherpaModelInfo {
  name: string;
  path: string;
  size_mb: number;
  quantization: QuantizationType;
  speed: string;
  status: ModelStatus;
  description: string;
}

export function formatFileSize(sizeMb: number): string {
  if (sizeMb >= 1024) return `${(sizeMb / 1024).toFixed(1)} GB`;
  return `${Math.round(sizeMb)} MB`;
}

export const SherpaAPI = {
  async init(): Promise<void> {
    await invoke('sherpa_init');
  },

  async getAvailableModels(): Promise<SherpaModelInfo[]> {
    return await invoke('sherpa_get_available_models');
  },

  async loadModel(modelName: string): Promise<void> {
    await invoke('sherpa_load_model', { modelName });
  },

  async downloadModel(modelName: string): Promise<void> {
    await invoke('sherpa_download_model', { modelName });
  },

  async deleteModel(modelName: string): Promise<string> {
    return await invoke('sherpa_delete_model', { modelName });
  },

  async getExecutionProvider(): Promise<string> {
    return await invoke('sherpa_get_execution_provider');
  },

  async setExecutionProvider(provider: string): Promise<void> {
    await invoke('sherpa_set_execution_provider', { provider });
  },

  async openModelsFolder(): Promise<void> {
    await invoke('open_sherpa_models_folder');
  }
};
