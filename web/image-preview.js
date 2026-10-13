import {installDialogFocus} from './vendor/promptstudio/dialog-focus.js';

// One viewer for the workspace and its embedded training monitor.
export function openImagePreview(src, alt = 'Image preview', trigger = document.activeElement) {
  if (!src) return;
  try {
    if (window.parent !== window && window.parent.loraStudioImagePreview) {
      window.parent.loraStudioImagePreview(src, alt, trigger);
      return;
    }
  } catch { /* A standalone monitor can also be embedded by another origin. */ }
  const existing = document.querySelector('.image-lightbox');
  if (existing) return;
  const dialog = document.createElement('dialog');
  dialog.className = 'image-lightbox';
  dialog.setAttribute('aria-label', alt || 'Image preview');
  dialog.innerHTML = '<div class="image-lightbox-actions"><a class="button" download>Download image</a><button type="button" aria-label="Close image preview" autofocus>✕</button></div><img><p class="image-lightbox-error" role="status" hidden>Unable to load this image. Close the preview and try again.</p>';
  const image = dialog.querySelector('img');
  image.alt = alt;
  image.addEventListener('error', () => { dialog.querySelector('.image-lightbox-error').hidden = false; });
  image.src = src;
  dialog.querySelector('a').href = src;
  dialog.querySelector('button').addEventListener('click', () => dialog.close());
  let backdropPress = false;
  dialog.addEventListener('pointerdown', event => { backdropPress = event.target === dialog; });
  dialog.addEventListener('click', event => {
    if (event.target === dialog && backdropPress) dialog.close();
  });
  document.body.append(dialog);
  installDialogFocus(dialog);
  dialog.addEventListener('close', () => {
    dialog.remove();
    image.removeAttribute('src');
    if (trigger?.isConnected) trigger.focus({preventScroll:true});
  });
  dialog.showModal();
}

window.loraStudioImagePreview = openImagePreview;
document.addEventListener('click', event => {
  const trigger = event.target.closest('[data-image-preview]');
  if (!trigger || trigger.disabled) return;
  const image = trigger.querySelector('img');
  const src = trigger.dataset.imagePreview || image?.currentSrc || image?.src;
  if (!src) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  openImagePreview(src, image?.alt || trigger.getAttribute('aria-label') || 'Image preview', trigger);
}, true);
