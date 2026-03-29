// Global utility functions

// Format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Format number
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function _resolveNode(element) {
    if (element == null) return null;
    return typeof element === 'string' ? document.querySelector(element) : element;
}

// Show loading spinner (no args: product-management forecast panels; with arg: single target)
function showLoading(element) {
    if (arguments.length === 0 || element === undefined || element === null) {
        var panelIds = ['demandForecast', 'salesForecast', 'profitForecast', 'inventoryManagement', 'stockOptimization'];
        panelIds.forEach(function (id) {
            var el = document.getElementById(id);
            if (el && el.innerHTML.indexOf('Select a product') === -1) {
                el.innerHTML = '<div class="text-center"><div class="loading-spinner"></div><br>Loading...</div>';
            }
        });
        return;
    }
    var spinner = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';
    var node = _resolveNode(element);
    if (node) node.innerHTML = spinner;
}

// Hide loading spinner
function hideLoading(element, content) {
    if (arguments.length === 0) return;
    var node = _resolveNode(element);
    if (node && content !== undefined) node.innerHTML = content;
}

// Toast notification
function showToast(message, type) {
    if (type === undefined) type = 'success';
    var toastHtml =
        '<div class="toast align-items-center text-white bg-' + type + ' border-0 position-fixed bottom-0 end-0 m-3" role="alert">' +
        '<div class="d-flex">' +
        '<div class="toast-body">' + message + '</div>' +
        '<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>' +
        '</div></div>';
    document.body.insertAdjacentHTML('beforeend', toastHtml);
    var toasts = document.querySelectorAll('.toast');
    var toastEl = toasts[toasts.length - 1];
    var toast = new bootstrap.Toast(toastEl);
    toast.show();
    toastEl.addEventListener('hidden.bs.toast', function () {
        toastEl.remove();
    }, { once: true });
}

// API calls
const api = {
    predict: async (data) => {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        return await response.json();
    },
    
    optimizeInventory: async (data) => {
        const response = await fetch('/api/optimize-inventory', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        return await response.json();
    },
    
    optimizeProfit: async (data) => {
        const response = await fetch('/api/optimize-profit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        return await response.json();
    },
    
    getSalesForecast: async (days = 7) => {
        const response = await fetch(`/api/sales-forecast?days=${days}`);
        return await response.json();
    }
};

// Auto-refresh data every 5 minutes if on dashboard
if (window.location.pathname.includes('dashboard')) {
    setInterval(() => {
        location.reload();
    }, 300000);
}

// Add tooltips
document.addEventListener('DOMContentLoaded', function() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Product Management: Analyze Product — explicit listener + /api/analyze_product kickoff
(function initProductAnalyzeButton() {
    var btn = document.getElementById('btnAnalyzeProduct');
    if (!btn) return;
    btn.addEventListener('click', async function (e) {
        e.preventDefault();
        console.log('[Analyze] button click');
        var select = document.getElementById('productSelect');
        var raw = select && select.value;
        var pid = raw ? parseInt(raw, 10) : NaN;
        if (!Number.isNaN(pid)) {
            console.log('[Analyze] product_id captured', pid);
            try {
                console.log('[Analyze] sending POST /api/analyze_product', { product_id: pid });
                var res = await fetch('/api/analyze_product', {
                    method: 'POST',
                    credentials: 'same-origin',
                    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
                    body: JSON.stringify({ product_id: pid }),
                });
                var body = await res.json().catch(function () { return {}; });
                console.log('[Analyze] response received', res.status, body);
            } catch (err) {
                console.error('[Analyze] /api/analyze_product failed', err);
            }
        } else {
            console.log('[Analyze] no dropdown product_id (custom or empty selection)');
        }
        if (typeof window.analyzeCustomProduct === 'function') {
            await window.analyzeCustomProduct();
        } else {
            console.error('[Analyze] analyzeCustomProduct is not defined');
        }
    });
})();