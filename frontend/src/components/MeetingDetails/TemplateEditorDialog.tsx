"use client";

import { useEffect, useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import { Copy, Plus, Save, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { SummaryTemplateInfo } from '@/hooks/meeting-details/useTemplates';

interface TemplateEditorDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  templates: SummaryTemplateInfo[];
  selectedTemplate: string;
  onTemplateSelect: (templateId: string, templateName: string) => void;
  onTemplatesChanged: () => Promise<SummaryTemplateInfo[]>;
}

const EMPTY_TEMPLATE = JSON.stringify({
  name: 'New Summary Template',
  description: 'Describe when this template should be used.',
  sections: [
    {
      title: 'Summary',
      instruction: 'Provide a concise summary of the meeting.',
      format: 'paragraph'
    },
    {
      title: 'Action Items',
      instruction: 'List clear action items with owners and deadlines when available.',
      format: 'list',
      item_format: '- [ ] Action item — Owner — Due date'
    }
  ]
}, null, 2);

function slugifyTemplateId(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 64);
}

function getTemplateName(templateJson: string): string | null {
  try {
    const parsed = JSON.parse(templateJson) as { name?: string };
    return parsed.name || null;
  } catch {
    return null;
  }
}

export function TemplateEditorDialog({
  open,
  onOpenChange,
  templates,
  selectedTemplate,
  onTemplateSelect,
  onTemplatesChanged,
}: TemplateEditorDialogProps) {
  const [editingTemplateId, setEditingTemplateId] = useState(selectedTemplate);
  const [templateId, setTemplateId] = useState(selectedTemplate);
  const [templateJson, setTemplateJson] = useState(EMPTY_TEMPLATE);
  const [validationMessage, setValidationMessage] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const editingTemplate = useMemo(
    () => templates.find((template) => template.id === editingTemplateId),
    [templates, editingTemplateId]
  );

  const loadTemplate = async (id: string) => {
    setIsLoading(true);
    setValidationMessage('');
    try {
      const json = await invoke<string>('api_get_template_json', { templateId: id });
      setEditingTemplateId(id);
      setTemplateId(id);
      setTemplateJson(json);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      toast.error('Failed to load template', { description: message });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      loadTemplate(selectedTemplate);
    }
  }, [open, selectedTemplate]);

  const handleNew = () => {
    setEditingTemplateId('');
    setTemplateId('new_summary_template');
    setTemplateJson(EMPTY_TEMPLATE);
    setValidationMessage('');
  };

  const handleDuplicate = () => {
    const name = getTemplateName(templateJson) || editingTemplate?.name || 'custom_template';
    setEditingTemplateId('');
    setTemplateId(slugifyTemplateId(`${name}_copy`) || 'custom_template_copy');
    setValidationMessage('');
  };

  const handleValidate = async () => {
    try {
      const name = await invoke<string>('api_validate_template', { templateJson });
      setValidationMessage(`Valid template: ${name}`);
      toast.success('Template is valid');
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setValidationMessage(message);
      toast.error('Template is invalid', { description: message });
      return false;
    }
  };

  const handleSave = async () => {
    const id = slugifyTemplateId(templateId);
    if (!id) {
      toast.error('Template ID is required');
      return;
    }

    setIsSaving(true);
    try {
      const saved = await invoke<SummaryTemplateInfo>('api_save_template', {
        templateId: id,
        templateJson,
      });
      setTemplateId(saved.id);
      setEditingTemplateId(saved.id);
      await onTemplatesChanged();
      onTemplateSelect(saved.id, saved.name);
      toast.success('Template saved', { description: saved.name });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      toast.error('Failed to save template', { description: message });
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async () => {
    if (!editingTemplate?.is_custom) {
      toast.error('Only custom templates can be deleted');
      return;
    }

    try {
      await invoke('api_delete_template', { templateId: editingTemplate.id });
      const refreshed = await onTemplatesChanged();
      const fallback = refreshed.find((template) => template.id === 'standard_meeting') || refreshed[0];
      if (fallback) {
        onTemplateSelect(fallback.id, fallback.name);
        await loadTemplate(fallback.id);
      }
      toast.success('Template deleted');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      toast.error('Failed to delete template', { description: message });
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-4xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Template Editor</DialogTitle>
        </DialogHeader>

        <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
          <div className="space-y-3 rounded-lg border bg-gray-50 p-3">
            <div className="flex items-center justify-between gap-2">
              <Label>Templates</Label>
              <Button type="button" variant="outline" size="sm" onClick={handleNew}>
                <Plus className="h-4 w-4" />
                New
              </Button>
            </div>

            <div className="max-h-[420px] space-y-2 overflow-y-auto pr-1">
              {templates.map((template) => (
                <button
                  key={template.id}
                  type="button"
                  onClick={() => loadTemplate(template.id)}
                  className={`w-full rounded-md border p-3 text-left transition ${editingTemplateId === template.id ? 'border-gray-900 bg-white shadow-sm' : 'border-gray-200 bg-white/70 hover:bg-white'}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-gray-900">{template.name}</span>
                    {template.is_custom && (
                      <span className="rounded-full bg-blue-50 px-2 py-0.5 text-[10px] font-medium text-blue-700">Custom</span>
                    )}
                  </div>
                  <div className="mt-1 font-mono text-[11px] text-gray-400">{template.id}</div>
                  <p className="mt-1 line-clamp-2 text-xs text-gray-500">{template.description}</p>
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
              <div className="space-y-2">
                <Label htmlFor="template-id">Template ID</Label>
                <Input
                  id="template-id"
                  value={templateId}
                  onChange={(event) => setTemplateId(slugifyTemplateId(event.target.value))}
                  placeholder="weekly_team_sync"
                />
                <p className="text-xs text-gray-500">Letters, numbers, hyphens, and underscores. Saving creates or replaces a custom template.</p>
              </div>
              <div className="flex items-end gap-2">
                <Button type="button" variant="outline" onClick={handleDuplicate} disabled={isLoading}>
                  <Copy className="h-4 w-4" />
                  Duplicate
                </Button>
                <Button type="button" variant="outline" onClick={handleDelete} disabled={!editingTemplate?.is_custom || isLoading}>
                  <Trash2 className="h-4 w-4" />
                  Delete
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="template-json">Template JSON</Label>
              <Textarea
                id="template-json"
                value={templateJson}
                onChange={(event) => setTemplateJson(event.target.value)}
                className="min-h-[420px] font-mono text-xs leading-relaxed"
                spellCheck={false}
              />
            </div>

            {validationMessage && (
              <div className="rounded-md border bg-gray-50 px-3 py-2 text-sm text-gray-700">{validationMessage}</div>
            )}

            <div className="flex flex-wrap justify-end gap-2">
              <Button type="button" variant="outline" onClick={handleValidate} disabled={isLoading || isSaving}>
                Validate
              </Button>
              <Button type="button" onClick={handleSave} disabled={isLoading || isSaving}>
                <Save className="h-4 w-4" />
                {isSaving ? 'Saving...' : 'Save template'}
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
