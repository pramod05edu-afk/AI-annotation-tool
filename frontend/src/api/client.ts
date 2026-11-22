/**
 * Typed API client. The types in schema.d.ts are GENERATED from the backend's OpenAPI schema
 * (`npm run gen:api`), so a wrong path, parameter or field is a compile error, not a runtime bug.
 */
import createClient from 'openapi-fetch'

import type { components, paths } from './schema'

export const api = createClient<paths>({ baseUrl: '' })

type Schemas = components['schemas']
export type Project = Schemas['ProjectRead']
export type ImageInfo = Schemas['ImageRead']
export type ImageStatus = Schemas['ImageStatus']
export type AnnotationIn = Schemas['AnnotationIn']
export type AnnotationRead = Schemas['AnnotationRead']
export type UploadResult = Schemas['UploadResult']
export type AiStatus = Schemas['AiStatus']
export type ModelSlotStatus = Schemas['ModelSlotStatus']

export const imageFileUrl = (imageId: number) => `/api/images/${imageId}/file`

/** Turn FastAPI's error body ({detail: "..."} or a validation list) into one readable sentence. */
export function errorMessage(error: unknown): string {
  if (error && typeof error === 'object' && 'detail' in error) {
    const detail = (error as { detail: unknown }).detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map(
          (d: { msg?: string; loc?: unknown[] }) => `${d.loc?.slice(1).join('.') ?? ''}: ${d.msg}`,
        )
        .join('; ')
    }
  }
  if (error instanceof Error) return error.message
  return 'Something went wrong'
}

/** Unwrap an openapi-fetch result: return the data or throw a readable Error. */
export function unwrap<T>(result: { data?: T; error?: unknown; response: Response }): T {
  if (result.error !== undefined || result.data === undefined) {
    if (result.response.status === 204) return undefined as T
    throw new Error(errorMessage(result.error ?? `HTTP ${result.response.status}`))
  }
  return result.data
}

/**
 * Upload files in batches (the API accepts at most 200 per request; smaller batches give a
 * smoother progress bar). Uses fetch directly because the body is multipart form data.
 */
export async function uploadImages(
  projectId: number,
  files: File[],
  onProgress: (done: number, total: number) => void,
  batchSize = 20,
): Promise<UploadResult> {
  const result: UploadResult = { uploaded: [], rejected: [] }
  for (let start = 0; start < files.length; start += batchSize) {
    const form = new FormData()
    for (const file of files.slice(start, start + batchSize)) form.append('files', file)
    const response = await fetch(`/api/projects/${projectId}/images`, {
      method: 'POST',
      body: form,
    })
    const body: unknown = await response.json().catch(() => undefined)
    if (!response.ok) throw new Error(errorMessage(body ?? `HTTP ${response.status}`))
    const batch = body as UploadResult
    result.uploaded.push(...batch.uploaded)
    result.rejected.push(...batch.rejected)
    onProgress(Math.min(start + batchSize, files.length), files.length)
  }
  return result
}
