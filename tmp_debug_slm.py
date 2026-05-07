from api.server_v3 import _load_system
sys_ = _load_system()
inputs = {'soil_type': 'Sandy', 'ph': 6.5, 'soil_moisture': 30.0, 'organic_carbon': 0.8, 'ec': 0.5, 'N': 50.0, 'P': 40.0, 'K': 50.0, 'temperature': 36.0, 'humidity': 70.0, 'rainfall': 120.0, 'season': 'Zaid', 'irrigation': 'Sprinkler', 'region': 'East', 'previous_crop': 'Maize', 'farm_size': 3.0}

# Direct model test
advisory_engine = sys_["gen"].advisory_engine
predictions = advisory_engine.predictor.predict(inputs)
print("Predictions:", predictions['crop'], predictions['crop_confidence'])

# Test the prompt build
prompt = advisory_engine._build_advisory_prompt(predictions, "Rice grows in flooded conditions.")
print("\n--- PROMPT START ---")
print(prompt[:600])
print("\n--- PROMPT END ---")
print(prompt[-300:])

# Test direct SLM generation
if advisory_engine.model:
    ids = advisory_engine.tokenizer.encode_prompt(prompt)
    print(f"\nToken count: {len(ids)}")
    print("First 10 tokens:", ids[:10])
    print("Last 10 tokens:", ids[-10:])
    
    # Try to see what the model outputs
    gen_ids = advisory_engine.model.generate(ids, max_tokens=100, temperature=0.55)
    print(f"\nGenerated {len(gen_ids)} tokens")
    raw_output = advisory_engine.tokenizer.decode(gen_ids, skip_special=True)
    print(f"Raw output ({len(raw_output)} chars):")
    print(repr(raw_output[:200]))
else:
    print("\nModel is None!")
