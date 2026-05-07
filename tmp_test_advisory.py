from api.server_v3 import _load_system
sys_ = _load_system()
inputs = {
    'soil_type': 'Sandy',
    'ph': 6.5,
    'soil_moisture': 30.0,
    'organic_carbon': 0.8,
    'ec': 0.5,
    'N': 50.0,
    'P': 40.0,
    'K': 50.0,
    'temperature': 36.0,
    'humidity': 70.0,
    'rainfall': 120.0,
    'season': 'Zaid',
    'irrigation': 'Sprinkler',
    'region': 'East',
    'previous_crop': 'Maize',
    'farm_size': 3.0,
}
result = sys_["gen"].advisory_engine.generate_advisory(inputs)
print('SOURCE:', result['source'])
print('RAG CONTEXT LEN:', len(result['rag_context']))
print('ADVISORY TEXT START:')
print(result['advisory_text'][:800])
print('---END---')
