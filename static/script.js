let traceOpen = false;
let library = { folders: [], papers: [] };
let currentFolder = 'all';
let pendingSavePaper = null;

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    tab.classList.add('active');
    const viewId = tab.dataset.tab + 'View';
    document.getElementById(viewId).classList.add('active');
    if (tab.dataset.tab === 'library') {
      loadLibrary();
    }
  });
});

function toggleTrace() {
  traceOpen = !traceOpen;
  document.getElementById('traceEntries').classList.toggle('open', traceOpen);
  document.getElementById('traceArrow').innerHTML = traceOpen ? '&#9660;' : '&#9654;';
}

async function submitQuery() {
  const q = document.getElementById('question').value.trim();
  if (!q) return;

  document.getElementById('resultCard').classList.remove('active');
  document.getElementById('errorCard').classList.remove('active');
  document.getElementById('citedPapersSection').classList.remove('active');
  document.getElementById('loading').classList.add('active');
  document.getElementById('hopLog').innerHTML = '';
  document.getElementById('loadingText').textContent = 'Searching...';
  document.getElementById('submitBtn').disabled = true;

  const hopMessages = [
    'Expanding query...',
    'Searching vector index...',
    'Retrieving documents...',
    'Generating answer...',
  ];
  let hopIdx = 0;
  const hopInterval = setInterval(() => {
    if (hopIdx < hopMessages.length) {
      const entry = document.createElement('div');
      entry.className = 'hop-entry';
      entry.textContent = hopMessages[hopIdx++];
      document.getElementById('hopLog').appendChild(entry);
      document.getElementById('loadingText').textContent = 'Processing...';
    }
  }, 2500);

  try {
    const res = await fetch('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    clearInterval(hopInterval);
    const data = await res.json();

    if (data.error) {
      document.getElementById('errorCard').textContent = data.error;
      document.getElementById('errorCard').classList.add('active');
    } else {
      document.getElementById('resultBody').textContent = data.answer;
      document.getElementById('hopCount').textContent = `${data.hops} step${data.hops !== 1 ? 's' : ''}`;

      const dot = document.getElementById('statusDot');
      const text = document.getElementById('statusText');
      if (data.low_confidence) {
        dot.className = 'status-dot warn';
        text.textContent = 'Low confidence';
      } else {
        dot.className = 'status-dot';
        text.textContent = 'Answer';
      }

      const traceEntries = document.getElementById('traceEntries');
      traceEntries.innerHTML = '';
      if (data.trace && data.trace.length > 0) {
        data.trace.forEach(t => {
          const div = document.createElement('div');
          div.className = 'trace-hop';
          div.innerHTML = `
            <div class="trace-hop-header">Step ${t.hop} - ${t.tool}</div>
            <div class="trace-hop-body">
              Args: ${JSON.stringify(t.args)}<br>
              Result: ${t.result_summary}
            </div>`;
          traceEntries.appendChild(div);
        });
        document.getElementById('traceSection').classList.add('active');
      }

      if (data.cited_papers && data.cited_papers.length > 0) {
        renderCitedPapers(data.cited_papers);
      }

      document.getElementById('resultCard').classList.add('active');
    }
  } catch (e) {
    clearInterval(hopInterval);
    document.getElementById('errorCard').textContent = 'Request failed: ' + e.message;
    document.getElementById('errorCard').classList.add('active');
  }

  document.getElementById('loading').classList.remove('active');
  document.getElementById('submitBtn').disabled = false;
}

function renderCitedPapers(papers) {
  const list = document.getElementById('citedPapersList');
  list.innerHTML = '';
  
  papers.forEach(paper => {
    const inLibrary = library.papers.some(p => p.pmid === paper.pmid);
    const div = document.createElement('div');
    div.className = 'cited-paper';
    div.innerHTML = `
      <div class="cited-paper-info">
        <div class="cited-paper-title">${escapeHtml(paper.title)}</div>
        <div class="cited-paper-pmid">
          PMID: <a href="https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/" target="_blank">${paper.pmid}</a>
        </div>
      </div>
      <div class="cited-paper-actions">
        <button class="save-paper-btn" ${inLibrary ? 'disabled' : ''} onclick="openSavePaperModal('${paper.pmid}', '${escapeHtml(paper.title).replace(/'/g, "\\'")}')">
          ${inLibrary ? 'Saved' : 'Save'}
        </button>
      </div>
    `;
    list.appendChild(div);
  });
  
  document.getElementById('citedPapersSection').classList.add('active');
}

// Library functions
async function loadLibrary() {
  try {
    const res = await fetch('/library');
    library = await res.json();
    renderFolders();
    renderPapers();
  } catch (e) {
    console.error('Failed to load library:', e);
  }
}

function renderFolders() {
  const list = document.getElementById('folderList');
  list.innerHTML = `
    <div class="folder-item ${currentFolder === 'all' ? 'active' : ''}" data-folder-id="all" onclick="selectFolder('all')">All Papers</div>
    <div class="folder-item ${currentFolder === 'unfiled' ? 'active' : ''}" data-folder-id="unfiled" onclick="selectFolder('unfiled')">Unfiled</div>
  `;
  
  library.folders.forEach(folder => {
    const div = document.createElement('div');
    div.className = `folder-item ${currentFolder === folder.id ? 'active' : ''}`;
    div.dataset.folderId = folder.id;
    div.innerHTML = `
      <span onclick="selectFolder('${folder.id}')">${escapeHtml(folder.name)}</span>
      <button class="delete-folder" onclick="event.stopPropagation(); deleteFolder('${folder.id}')">x</button>
    `;
    list.appendChild(div);
  });
  
  updateFolderSelect();
}

function updateFolderSelect() {
  const select = document.getElementById('savePaperFolder');
  select.innerHTML = '<option value="">Unfiled</option>';
  library.folders.forEach(folder => {
    select.innerHTML += `<option value="${folder.id}">${escapeHtml(folder.name)}</option>`;
  });
}

function selectFolder(folderId) {
  currentFolder = folderId;
  renderFolders();
  renderPapers();
  
  let folderName = 'All Papers';
  if (folderId === 'unfiled') {
    folderName = 'Unfiled';
  } else if (folderId !== 'all') {
    const folder = library.folders.find(f => f.id === folderId);
    if (folder) folderName = folder.name;
  }
  document.getElementById('currentFolderName').textContent = folderName;
}

function renderPapers() {
  const list = document.getElementById('paperList');
  const emptyState = document.getElementById('emptyState');
  list.innerHTML = '';
  
  let papers = library.papers;
  if (currentFolder === 'unfiled') {
    papers = papers.filter(p => !p.folder_id);
  } else if (currentFolder !== 'all') {
    papers = papers.filter(p => p.folder_id === currentFolder);
  }
  
  document.getElementById('paperCount').textContent = `${papers.length} paper${papers.length !== 1 ? 's' : ''}`;
  
  if (papers.length === 0) {
    emptyState.classList.add('active');
    list.style.display = 'none';
  } else {
    emptyState.classList.remove('active');
    list.style.display = 'block';
    
    papers.forEach(paper => {
      const div = document.createElement('div');
      div.className = 'paper-card';
      div.innerHTML = `
        <div class="paper-card-info">
          <div class="paper-card-title">${escapeHtml(paper.title)}</div>
          <div class="paper-card-meta">
            <a href="https://pubmed.ncbi.nlm.nih.gov/${paper.pmid}/" target="_blank">PMID: ${paper.pmid}</a>
            <span>Saved ${formatDate(paper.saved_at)}</span>
          </div>
        </div>
        <div class="paper-card-actions">
          <select onchange="movePaper('${paper.id}', this.value)">
            <option value="" ${!paper.folder_id ? 'selected' : ''}>Unfiled</option>
            ${library.folders.map(f => `<option value="${f.id}" ${paper.folder_id === f.id ? 'selected' : ''}>${escapeHtml(f.name)}</option>`).join('')}
          </select>
          <button onclick="deletePaper('${paper.id}')">Remove</button>
        </div>
      `;
      list.appendChild(div);
    });
  }
}

// Folder Modal
document.getElementById('newFolderBtn').addEventListener('click', () => {
  document.getElementById('folderModal').classList.add('active');
  document.getElementById('folderNameInput').focus();
});

function closeFolderModal() {
  document.getElementById('folderModal').classList.remove('active');
  document.getElementById('folderNameInput').value = '';
}

async function createFolder() {
  const name = document.getElementById('folderNameInput').value.trim();
  if (!name) return;
  
  try {
    const res = await fetch('/library/folders', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    if (res.ok) {
      closeFolderModal();
      await loadLibrary();
    }
  } catch (e) {
    console.error('Failed to create folder:', e);
  }
}

async function deleteFolder(folderId) {
  if (!confirm('Delete this folder? Papers will be moved to Unfiled.')) return;
  
  try {
    await fetch(`/library/folders/${folderId}`, { method: 'DELETE' });
    if (currentFolder === folderId) {
      currentFolder = 'all';
    }
    await loadLibrary();
  } catch (e) {
    console.error('Failed to delete folder:', e);
  }
}

// Save Paper Modal
function openSavePaperModal(pmid, title) {
  pendingSavePaper = { pmid, title };
  document.getElementById('savePaperTitle').textContent = title;
  updateFolderSelect();
  document.getElementById('savePaperFolder').value = '';
  document.getElementById('savePaperModal').classList.add('active');
}

function closeSavePaperModal() {
  document.getElementById('savePaperModal').classList.remove('active');
  pendingSavePaper = null;
}

async function confirmSavePaper() {
  if (!pendingSavePaper) return;
  
  const folderId = document.getElementById('savePaperFolder').value || null;
  
  try {
    const res = await fetch('/library/papers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pmid: pendingSavePaper.pmid,
        title: pendingSavePaper.title,
        folder_id: folderId
      })
    });
    
    if (res.ok) {
      closeSavePaperModal();
      await loadLibrary();
      const citedPapers = document.querySelectorAll('.cited-paper');
      citedPapers.forEach(cp => {
        const pmidEl = cp.querySelector('.cited-paper-pmid a');
        if (pmidEl && pmidEl.textContent === pendingSavePaper.pmid) {
          const btn = cp.querySelector('.save-paper-btn');
          if (btn) {
            btn.disabled = true;
            btn.textContent = 'Saved';
          }
        }
      });
    } else {
      const data = await res.json();
      alert(data.error || 'Failed to save paper');
    }
  } catch (e) {
    console.error('Failed to save paper:', e);
  }
}

async function movePaper(paperId, folderId) {
  try {
    await fetch(`/library/papers/${paperId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ folder_id: folderId || null })
    });
    await loadLibrary();
  } catch (e) {
    console.error('Failed to move paper:', e);
  }
}

async function deletePaper(paperId) {
  if (!confirm('Remove this paper from your library?')) return;
  
  try {
    await fetch(`/library/papers/${paperId}`, { method: 'DELETE' });
    await loadLibrary();
  } catch (e) {
    console.error('Failed to delete paper:', e);
  }
}

// Utilities
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function formatDate(isoString) {
  const date = new Date(isoString);
  return date.toLocaleDateString();
}

// Keyboard shortcuts
document.getElementById('question').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submitQuery();
});

document.getElementById('folderNameInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') createFolder();
  if (e.key === 'Escape') closeFolderModal();
});

// Close modals on background click
document.getElementById('folderModal').addEventListener('click', e => {
  if (e.target.id === 'folderModal') closeFolderModal();
});

document.getElementById('savePaperModal').addEventListener('click', e => {
  if (e.target.id === 'savePaperModal') closeSavePaperModal();
});

// Initial load
loadLibrary();
