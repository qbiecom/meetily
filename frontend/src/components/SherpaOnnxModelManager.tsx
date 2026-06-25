import { useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import { SherpaAPI, SherpaModelInfo, ModelStatus, formatFileSize } from '../lib/sherpa';

interface SherpaOnnxModelManagerProps {
  selectedModel?: string;
  onModelSelect?: (modelName: string) => void;
  autoSave?: boolean;
}

export function SherpaOnnxModelManager({ selectedModel, onModelSelect, autoSave = false }: SherpaOnnxModelManagerProps) {
  const [models, setModels] = useState<SherpaModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [executionProvider, setExecutionProvider] = useState('cpu');

  const refreshModels = async () => {
    setLoading(true);
    try {
      await SherpaAPI.init();
      const [modelList, provider] = await Promise.all([
        SherpaAPI.getAvailableModels(),
        SherpaAPI.getExecutionProvider()
      ]);
      setModels(modelList);
      setExecutionProvider(provider);
    } catch (err) {
      toast.error('Failed to load Sherpa ONNX models', {
        description: err instanceof Error ? err.message : String(err)
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshModels();
  }, []);

  useEffect(() => {
    const unlisteners: Array<() => void> = [];

    const setup = async () => {
      unlisteners.push(await listen<{ modelName: string; progress: number }>('sherpa-model-download-progress', (event) => {
        const { modelName, progress } = event.payload;
        setModels(prev => prev.map(model =>
          model.name === modelName ? { ...model, status: { Downloading: { progress } } as ModelStatus } : model
        ));
      }));

      unlisteners.push(await listen<{ modelName: string }>('sherpa-model-download-complete', (event) => {
        const { modelName } = event.payload;
        setModels(prev => prev.map(model =>
          model.name === modelName ? { ...model, status: 'Available' as ModelStatus } : model
        ));
        toast.success(`${modelName} ready`);
      }));

      unlisteners.push(await listen<{ modelName: string; error: string }>('sherpa-model-download-error', (event) => {
        const { modelName, error } = event.payload;
        setModels(prev => prev.map(model =>
          model.name === modelName ? { ...model, status: { Error: error } as ModelStatus } : model
        ));
        toast.error(`Failed to download ${modelName}`, { description: error });
      }));
    };

    setup();
    return () => unlisteners.forEach(unlisten => unlisten());
  }, []);

  const handleProviderChange = async (provider: string) => {
    const previousProvider = executionProvider;
    setExecutionProvider(provider);
    try {
      await SherpaAPI.setExecutionProvider(provider);
      toast.info(`Sherpa ONNX will use ${provider.toUpperCase()}`);
    } catch (err) {
      setExecutionProvider(previousProvider);
      toast.error('Failed to update Sherpa ONNX execution provider', {
        description: err instanceof Error ? err.message : String(err)
      });
    }
  };

  const downloadModel = async (modelName: string) => {
    setModels(prev => prev.map(model =>
      model.name === modelName ? { ...model, status: { Downloading: { progress: 0 } } as ModelStatus } : model
    ));
    toast.info(`Downloading ${modelName}...`);
    await SherpaAPI.downloadModel(modelName);
  };

  const saveModelSelection = async (modelName: string) => {
    try {
      await invoke('api_save_transcript_config', {
        provider: 'sherpaOnnx',
        model: modelName,
        apiKey: null
      });
    } catch (error) {
      console.error('Failed to save Sherpa ONNX model selection:', error);
    }
  };

  const selectModel = async (modelName: string) => {
    await SherpaAPI.loadModel(modelName);
    onModelSelect?.(modelName);
    if (autoSave) {
      await saveModelSelection(modelName);
    }
    toast.success(`Switched to ${modelName}`);
  };

  const deleteModel = async (modelName: string) => {
    await SherpaAPI.deleteModel(modelName);
    await refreshModels();
    toast.success(`${modelName} deleted`);
  };

  if (loading) {
    return <div className="rounded-lg border border-gray-200 p-4 text-sm text-gray-600">Loading Sherpa ONNX models...</div>;
  }

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-blue-200 bg-blue-50 p-3">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="font-medium text-gray-900">Execution Provider</div>
            <div className="text-xs text-gray-600">CUDA requires the Sherpa ONNX CUDA runtime package at build/package time.</div>
          </div>
          <select
            value={executionProvider}
            onChange={(event) => handleProviderChange(event.target.value)}
            className="rounded-md border border-gray-300 bg-white px-3 py-2 text-sm"
          >
            <option value="cpu">CPU</option>
            <option value="cuda">CUDA</option>
          </select>
        </div>
      </div>

      {models.map(model => {
        const isAvailable = model.status === 'Available';
        const isMissing = model.status === 'Missing';
        const isSelected = selectedModel === model.name;
        const downloadProgress = typeof model.status === 'object' && 'Downloading' in model.status
          ? model.status.Downloading.progress
          : null;
        const isError = typeof model.status === 'object' && 'Error' in model.status;
        const isCorrupted = typeof model.status === 'object' && 'Corrupted' in model.status;

        return (
          <div
            key={model.name}
            onClick={() => isAvailable && selectModel(model.name)}
            className={`rounded-lg border-2 p-4 ${isSelected && isAvailable ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white'} ${isAvailable ? 'cursor-pointer hover:border-gray-300' : ''}`}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="font-mono text-sm font-semibold text-gray-900 sm:text-base">{model.name}</h3>
                  {isSelected && isAvailable && <span className="rounded-full bg-blue-600 px-2 py-0.5 text-xs font-medium text-white">Selected</span>}
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">Sherpa ONNX</span>
                  <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">{model.quantization}</span>
                  <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">{formatFileSize(model.size_mb)}</span>
                  <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">{model.speed}</span>
                </div>
                <p className="mt-2 text-sm text-gray-600">{model.description}</p>
              </div>

              <div className="flex shrink-0 items-center gap-2">
                {isAvailable && <span className="text-xs font-medium text-green-600">Ready</span>}
                {isAvailable && (
                  <button onClick={(event) => { event.stopPropagation(); deleteModel(model.name); }} className="rounded-md px-2 py-1 text-xs text-gray-500 hover:bg-red-50 hover:text-red-600">Delete</button>
                )}
                {(isMissing || isError || isCorrupted) && (
                  <button onClick={(event) => { event.stopPropagation(); downloadModel(model.name); }} className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700">
                    {isMissing ? 'Download' : 'Re-download'}
                  </button>
                )}
              </div>
            </div>

            {downloadProgress !== null && (
              <div className="mt-3 border-t border-gray-200 pt-3">
                <div className="mb-2 flex justify-between text-sm font-medium text-blue-600">
                  <span>Downloading...</span>
                  <span>{Math.round(downloadProgress)}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-gray-200">
                  <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${downloadProgress}%` }} />
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
