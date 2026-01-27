#!/usr/bin/env python3
"""
Teste da API Key do Gemini
"""
import sys

# Carregar API key
try:
    import toml
    with open('secrets.toml', 'r') as f:
        secrets = toml.load(f)
    api_key = secrets.get('GEMINI_API_KEY')
    print(f"✅ API Key carregada: {api_key[:20]}...")
except Exception as e:
    print(f"❌ Erro ao carregar API Key: {e}")
    sys.exit(1)

# Testar google.genai (novo)
print("\n" + "="*50)
print("TESTE 1: google-genai (nova biblioteca)")
print("="*50)
try:
    from google import genai
    print("✅ google.genai importado com sucesso")
    
    try:
        client = genai.Client(api_key=api_key)
        print("✅ Cliente criado com sucesso")
        
        # Testar geração de conteúdo
        test_prompt = "Say 'Hello, I am working!' in one sentence."
        print(f"\n🧪 Testando com prompt: {test_prompt}")
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=test_prompt
        )
        print(f"✅ Resposta recebida: {response.text}")
        
    except Exception as e:
        print(f"❌ Erro ao usar cliente: {type(e).__name__}: {e}")
        
except ImportError as e:
    print(f"❌ google.genai não instalado: {e}")
except Exception as e:
    print(f"❌ Erro inesperado: {type(e).__name__}: {e}")

# Testar google.generativeai (antigo)
print("\n" + "="*50)
print("TESTE 2: google-generativeai (biblioteca antiga)")
print("="*50)
try:
    import google.generativeai as genai_old
    print("✅ google.generativeai importado com sucesso")
    
    try:
        genai_old.configure(api_key=api_key)
        print("✅ API configurada com sucesso")
        
        # Testar geração de conteúdo
        model = genai_old.GenerativeModel("gemini-2.0-flash-exp")
        print("✅ Modelo criado com sucesso")
        
        test_prompt = "Say 'Hello, I am working!' in one sentence."
        print(f"\n🧪 Testando com prompt: {test_prompt}")
        
        response = model.generate_content(test_prompt)
        print(f"✅ Resposta recebida: {response.text}")
        
    except Exception as e:
        print(f"❌ Erro ao usar biblioteca antiga: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"❌ google.generativeai não instalado: {e}")
except Exception as e:
    print(f"❌ Erro inesperado: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("TESTE CONCLUÍDO")
print("="*50)
