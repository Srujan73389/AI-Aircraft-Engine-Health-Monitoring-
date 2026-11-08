import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // 1. Health check
  checkHealth: async () => {
    try {
      const response = await axios.get('/health', { timeout: 3000 });
      return response.data;
    } catch (err) {
      // Fallback check against API root
      const fallback = await apiClient.get('/fleet/subsets', { timeout: 3000 });
      return { status: 'healthy', fallback: true };
    }
  },

  // 2. Fleet & Engine Discovery
  getFleetSubsets: async () => {
    const response = await apiClient.get('/fleet/subsets');
    return response.data;
  },

  getSubsetEngines: async (subset) => {
    const response = await apiClient.get(`/fleet/${subset}/engines`);
    return response.data;
  },

  getEngineCycleState: async (subset, unit, cycle) => {
    const response = await apiClient.get(`/engine/${subset}/${unit}/cycle/${cycle}`);
    return response.data;
  },

  getEngineSensors: async (subset, unit) => {
    const response = await apiClient.get(`/engine/${subset}/${unit}/sensors`);
    return response.data;
  },

  // 3. Monitoring & Anomaly Detection
  checkAnomaly: async (payload) => {
    const response = await apiClient.post('/monitoring/anomaly-check', payload);
    return response.data;
  },

  // 4. Prognostics & RUL
  predictRUL: async (payload) => {
    const response = await apiClient.post('/predict/rul', payload);
    return response.data;
  },

  // 5. Simulation What-If
  runSimulationWhatIf: async (payload) => {
    const response = await apiClient.post('/simulation/what-if', payload);
    return response.data;
  },

  // 6. Safety Verification & Governance
  verifySafetyAction: async (payload) => {
    const response = await apiClient.post('/safety/verify', payload);
    return response.data;
  },

  executeGatedAction: async (payload) => {
    const response = await apiClient.post('/safety/execute-gated', payload);
    return response.data;
  },

  getAuditTrail: async (params = {}) => {
    const response = await apiClient.get('/safety/audit-trail', { params });
    return response.data;
  },
};

export default apiService;
