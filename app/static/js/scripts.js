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

// Show loading spinner
function showLoading(element) {
    const spinner = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';
    $(element).html(spinner);
}

// Hide loading spinner
function hideLoading(element, content) {
    $(element).html(content);
}

// Toast notification
function showToast(message, type = 'success') {
    const toastHtml = `
        <div class="toast align-items-center text-white bg-${type} border-0 position-fixed bottom-0 end-0 m-3" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;
    
    $('body').append(toastHtml);
    const toast = new bootstrap.Toast($('.toast').last()[0]);
    toast.show();
    
    // Remove toast after it's hidden
    $('.toast').last().on('hidden.bs.toast', function() {
        $(this).remove();
    });
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