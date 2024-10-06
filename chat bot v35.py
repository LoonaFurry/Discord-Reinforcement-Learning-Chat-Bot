import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import aiohttp
import aiosqlite
import backoff
import discord
import networkx as nx
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from groq import Groq
from transitions import Machine
from duckduckgo_search import AsyncDDGS
from langdetect import detect, LangDetectException
import functools
from discord.ext import commands, tasks

HUGGINGFACE_TOKEN = "huggingface-token-here"  # Replace with your Hugging Face token!
login(HUGGINGFACE_TOKEN)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler("hata.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)])

intents = discord.Intents.all()
intents.message_content = True
intents.members = True
bot = discord.Client(intents=intents)
bot = commands.Bot(command_prefix="!", intents=intents) 

discord_token = "your-discord-token"
groq_api_key = "groq-api key"

embedding_model = SentenceTransformer('all-mpnet-base-v2')

KOD_DİZİNİ = os.path.dirname(__file__)
VERİTABANI_DOSYASI = os.path.join(KOD_DİZİNİ, "sohbet_gecmisi.db")
KULLANICI_PROFİLLERİ_DOSYASI = os.path.join(KOD_DİZİNİ, "kullanici_profilleri.json")
BİLGİ_GRAFİĞİ_DOSYASI = os.path.join(KOD_DİZİNİ, "bilgi_grafi.pkl")

BAĞLAM_PENCERESİ_BOYUTU = 8000
kullanıcı_profilleri = defaultdict(lambda: {"tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []}, "demografi": {"yaş": None, "konum": None}, "geçmiş_özeti": "", "bağlam": deque(maxlen=BAĞLAM_PENCERESİ_BOYUTU), "kişilik": {"mizah": 0.5, "nezaket": 0.8, "iddialılık": 0.6, "yaratıcılık": 0.5}, "diyalog_durumu": "karşılama", "uzun_süreli_hafıza": [], "son_bot_eylemi": None, "ilgiler": [], "sorgu": "", "planlama_durumu": {}, "etkileşim_geçmişi": [], "feedback_topics": [], "feedback_keywords": [], "satisfaction": 0, "duygusal_durum": "nötr", "çıkarımlar": []})

DİYALOG_DURUMLARI = ["karşılama", "soru_cevap", "hikaye_anlatma", "genel_konuşma", "planlama", "çıkış_durumu"]
BOT_EYLEMLERİ = ["bilgilendirici_yanıt", "yaratıcı_yanıt", "açıklayıcı_soru", "diyalog_durumunu_değiştir", "yeni_konu_başlat", "plan_oluştur", "planı_uygula"]

tfidf_vektörleştirici = TfidfVectorizer()
model = SentenceTransformer('all-mpnet-base-v2')

hata_sayacı = 0
aktif_kullanıcılar = 0
yanıt_süresi_histogramı = []
yanıt_süresi_özeti = []
geri_bildirim_sayısı = 0
veritabanı_hazır = False
veritabanı_kilidi = asyncio.Lock()


class BilgiGrafiği:
    def __init__(self):
        self.grafik = nx.DiGraph()
        self.düğüm_kimliği_sayacı = 0

    def _düğüm_kimliği_oluştur(self):
        self.düğüm_kimliği_sayacı += 1
        return str(self.düğüm_kimliği_sayacı)

    def düğüm_ekle(self, düğüm_türü, düğüm_kimliği=None, veri=None):
        if düğüm_kimliği is None:
            düğüm_kimliği = self._düğüm_kimliği_oluştur()
        self.grafik.add_node(düğüm_kimliği, tür=düğüm_türü, veri=veri if veri is not None else {})

    def düğüm_al(self, düğüm_kimliği):
        return self.grafik.nodes.get(düğüm_kimliği)

    def kenar_ekle(self, kaynak_kimliği, ilişki, hedef_kimliği, özellikler=None):
        self.grafik.add_edge(kaynak_kimliği, hedef_kimliği, ilişki=ilişki, özellikler=özellikler if özellikler is not None else {})

    def ilgili_düğümleri_al(self, düğüm_kimliği, ilişki=None, yön="giden"):
        ilgili_düğümler = []
        if yön == "giden" or yön == "her ikisi":
            for komşu in self.grafik.successors(düğüm_kimliği):
                kenar_verisi = self.grafik.get_edge_data(düğüm_kimliği, komşu)
                if ilişki is None or kenar_verisi["ilişki"] == ilişki:
                    ilgili_düğümler.append(self.düğüm_al(komşu))
        if yön == "gelen" or yön == "her ikisi":
            for komşu in self.grafik.predecessors(düğüm_kimliği):
                kenar_verisi = self.grafik.get_edge_data(komşu, düğüm_kimliği)
                if ilişki is None or kenar_verisi["ilişki"] == ilişki:
                    ilgili_düğümler.append(self.düğüm_al(komşu))
        return ilgili_düğümler

    async def metni_göm(self, metin):
        embedding = model.encode(metin)
        return embedding.tolist()

    async def düğümleri_ara(self, sorgu, üst_k=3, düğüm_türü=None):
        sorgu_gömmesi = await self.metni_göm(sorgu)
        sonuçlar = []
        for düğüm_kimliği, düğüm_verisi in self.grafik.nodes(data=True):
            if düğüm_türü is None or düğüm_verisi["tür"] == düğüm_türü:
                düğüm_gömmesi = await self.metni_göm(str(düğüm_verisi["veri"]))
                benzerlik = cosine_similarity([sorgu_gömmesi], [düğüm_gömmesi])[0][0]
                sonuçlar.append((düğüm_verisi["tür"], düğüm_kimliği, düğüm_verisi["veri"], benzerlik))

        sonuçlar.sort(key=lambda x: x[3], reverse=True)
        return sonuçlar[:üst_k]

    def düğümü_güncelle(self, düğüm_kimliği, yeni_veri):
        self.grafik.nodes[düğüm_kimliği]["veri"].update(yeni_veri)

    def düğümü_sil(self, düğüm_kimliği):
        self.grafik.remove_node(düğüm_kimliği)

    def dosyaya_kaydet(self, dosya_yolu):
        nx.write_gpickle(self.grafik, dosya_yolu)

    @staticmethod
    def dosyadan_yükle(dosya_yolu):
        bg = BilgiGrafiği()
        bg.grafik = nx.read_gpickle(dosya_yolu)
        return bg


bilgi_grafiği = BilgiGrafiği()
if os.path.exists(BİLGİ_GRAFİĞİ_DOSYASI):
    bilgi_grafiği = BilgiGrafiği.dosyadan_yükle(BİLGİ_GRAFİĞİ_DOSYASI)


async def uzun_süreli_hafızaya_kaydet(kullanıcı_kimliği, bilgi_türü, bilgi):
    bilgi_grafiği.düğüm_ekle(bilgi_türü, veri={"kullanıcı_kimliği": kullanıcı_kimliği, "bilgi": bilgi})
    bilgi_grafiği.kenar_ekle(kullanıcı_kimliği, "sahiptir_" + bilgi_türü, str(bilgi_grafiği.düğüm_kimliği_sayacı - 1))
    bilgi_grafiği.dosyaya_kaydet(BİLGİ_GRAFİĞİ_DOSYASI)


async def uzun_süreli_hafızadan_al(kullanıcı_kimliği, bilgi_türü, sorgu=None, üst_k=3):
    if sorgu:
        arama_sonuçları = await bilgi_grafiği.düğümleri_ara(sorgu, üst_k=üst_k, düğüm_türü=bilgi_türü)
        return [(düğüm_türü, düğüm_kimliği, düğüm_verisi) for düğüm_türü, düğüm_kimliği, düğüm_verisi, skor in arama_sonuçları]
    else:
        ilgili_düğümler = bilgi_grafiği.ilgili_düğümleri_al(kullanıcı_kimliği, "sahiptir_" + bilgi_türü)
        return [düğüm["veri"]["bilgi"] for düğüm in ilgili_düğümler]


async def plan_adımını_yürüt(plan, adım_indeksi, kullanıcı_kimliği, mesaj):
    adım = plan["adımlar"][adım_indeksi]
    yürütme_istemi = f"You are an AI assistant helping a user carry out a plan. Here is the plan step: {adım['açıklama']} The user said: {mesaj.content} If the user's message indicates they are ready to proceed with this step, provide a simulated response as if they completed it. If the user requests clarification or changes, accept their request and provide helpful information or guidance. Be specific and relevant to the plan step."

    try:
        yürütme_yanıtı = await groq_ile_yanıt_oluştur(yürütme_istemi, kullanıcı_kimliği)
    except Exception as e:
        logging.error(f"Error occurred while executing plan step: {e}")
        return "An error occurred while trying to execute this step. Please try again later."


    adım["durum"] = "devam_ediyor"
    await uzun_süreli_hafızaya_kaydet(kullanıcı_kimliği, "plan_uygulama_sonucu", {"adım_açıklaması": adım["açıklama"], "sonuç": "devam_ediyor", "zaman_damgası": datetime.now(timezone.utc).isoformat()})
    return yürütme_yanıtı


async def plan_yürütmesini_izle(plan, kullanıcı_kimliği, mesaj):
    geçerli_adım_indeksi = next((i for i, adım in enumerate(plan["adımlar"]) if adım["durum"] == "devam_ediyor"), None)

    if geçerli_adım_indeksi is not None:
        if "bitti" in mesaj.content.lower() or "tamamlandı" in mesaj.content.lower():
            plan["adımlar"][geçerli_adım_indeksi]["durum"] = "tamamlandı"
            await mesaj.channel.send(f"Great! Step {geçerli_adım_indeksi + 1} has been completed.")
            if geçerli_adım_indeksi + 1 < len(plan["adımlar"]):
                sonraki_adım_yanıtı = await plan_adımını_yürüt(plan, geçerli_adım_indeksi + 1, kullanıcı_kimliği, mesaj)
                return f"Moving on to the next step: {sonraki_adım_yanıtı}"
            else:
                return "Congratulations! You have completed all the steps in the plan."
        else:
            return await plan_adımını_yürüt(plan, geçerli_adım_indeksi, kullanıcı_kimliği, mesaj)


async def plan_oluştur(hedef, tercihler, kullanıcı_kimliği, mesaj):
    planlama_istemi = f"You are an AI assistant specialized in planning. A user needs help with the following goal: {hedef} What the user said about the plan: {tercihler.get('kullanıcı_girdisi')} Based on this information, create a detailed and actionable plan by identifying key steps and considerations. Ensure the plan is: * **Specific:** Each step should be clearly defined. * **Measurable:** Add ways to track progress. * **Achievable:** Steps should be realistic and actionable. * **Relevant:** Align with the user's goal. * **Time-bound:** Include estimated timelines or deadlines. Analyze potential risks and dependencies for each step. Format the plan as a JSON object: ```json {{ 'hedef': 'User's goal', 'adımlar': [ {{ 'açıklama': 'Step description', 'son_tarih': 'Optional deadline for the step', 'bağımlılıklar': ['List of dependencies (other step descriptions)'], 'riskler': ['List of potential risks'], 'durum': 'waiting' }}, // ... more steps ], 'tercihler': {{ // User preferences related to the plan }} }} ```"
    try:
        plan_metni = await groq_ile_yanıt_oluştur(planlama_istemi, kullanıcı_kimliği)
        plan = json.loads(plan_metni)
    except (json.JSONDecodeError, Exception) as e:
        logging.error(f"Error occurred while parsing JSON or creating plan: {e}")
        return {"hedef": hedef, "adımlar": [], "tercihler": tercihler}

    await uzun_süreli_hafızaya_kaydet(kullanıcı_kimliği, "plan", plan)
    return plan


async def planı_değerlendir(plan, kullanıcı_kimliği):
    değerlendirme_istemi = f"You are an AI assistant tasked with evaluating a plan, including identifying potential risks and dependencies. Here is the plan: Goal: {plan['hedef']} Steps: {json.dumps(plan['adımlar'], indent=2)} Evaluate this plan based on the following criteria: * **Feasibility:** Is the plan realistically achievable? * **Completeness:** Does the plan cover all necessary steps? * **Efficiency:** Is the plan optimally structured? Are there unnecessary or redundant steps? * **Risks:** Analyze the risks identified for each step. Are they significant? How can they be mitigated? * **Dependencies:** Are the dependencies between steps clear and well defined? Are there potential conflicts or bottlenecks? * **Improvements:** Suggest any improvements or alternative approaches considering the risks and dependencies. Provide a structured evaluation summarizing your assessment for each criterion. Be as specific as possible in your analysis."
    try:
        değerlendirme_metni = await groq_ile_yanıt_oluştur(değerlendirme_istemi, kullanıcı_kimliği)
    except Exception as e:
        logging.error(f"Error occurred while evaluating plan: {e}")
        return {"değerlendirme_metni": "An error occurred while evaluating the plan. Please try again later."}

    await uzun_süreli_hafızaya_kaydet(kullanıcı_kimliği, "plan_değerlendirmesi", değerlendirme_metni)
    return {"değerlendirme_metni": değerlendirme_metni}



async def planı_doğrula(plan, kullanıcı_kimliği):
    doğrulama_istemi = f"You are an AI assistant specialized in evaluating the feasibility and safety of plans. Carefully analyze the following plan and identify any potential issues, flaws, or missing information that could lead to failure or undesirable outcomes. Goal: {plan['hedef']} Steps: {json.dumps(plan['adımlar'], indent=2)} Consider the following points: * **Clarity and Specificity:** Are the steps clear and specific enough to be actionable? * **Realism and Feasibility:** Are the steps realistic and achievable considering the user's context and resources? * **Dependencies:** Are the dependencies between steps clearly stated and logical? Are there cyclic dependencies? * **Time Constraints:** Are the deadlines realistic and achievable? Are there potential time conflicts? * **Resource Availability:** Are the necessary resources available for each step? * **Risk Assessment:** Are potential risks sufficiently identified and analyzed? Are there mitigation strategies? * **Safety and Ethics:** Does the plan comply with safety and ethical standards? Are there potential negative outcomes? Provide a detailed analysis of the plan highlighting any weaknesses or areas for improvement. Indicate if the plan is solid and well-structured, or provide specific recommendations for making it more robust and effective."

    try:
        doğrulama_sonucu = await groq_ile_yanıt_oluştur(doğrulama_istemi, kullanıcı_kimliği)
    except Exception as e:
        logging.error(f"Error occurred while validating plan: {e}")
        return False, "An error occurred while validating the plan. Please try again later."


    logging.info(f"Plan validation result: {doğrulama_sonucu}")

    if "valid" in doğrulama_sonucu.lower():
        return True, doğrulama_sonucu
    else:
        return False, doğrulama_sonucu



async def plan_geri_bildirimini_işle(kullanıcı_kimliği, mesaj):
    geri_bildirim_istemi = f"You are an AI assistant analyzing user feedback on a plan. The user said: {mesaj} Is the user accepting the plan? Respond with 'ACCEPT' if yes. If no, identify parts of the plan the user wants to change and suggest how the plan might be revised."
    try:
        geri_bildirim_analizi = await groq_ile_yanıt_oluştur(geri_bildirim_istemi, kullanıcı_kimliği)
        if "accept" in geri_bildirim_analizi.lower():
            return "accept"
        else:
            return geri_bildirim_analizi
    except Exception as e:
        logging.error(f"Error occurred while processing plan feedback: {e}")
        return "An error occurred while processing your feedback. Please try again later."



kullanıcı_mesaj_tamponu = defaultdict(list)


async def kullanıcı_ilgi_alanlarını_belirle(kullanıcı_kimliği, mesaj):
    kullanıcı_mesaj_tamponu[kullanıcı_kimliği].append(mesaj)
    if len(kullanıcı_mesaj_tamponu[kullanıcı_kimliği]) >= 5:
        mesajlar = kullanıcı_mesaj_tamponu[kullanıcı_kimliği]
        kullanıcı_mesaj_tamponu[kullanıcı_kimliği] = []
        gömmeler = [await bilgi_grafiği.metni_göm(mesaj) for mesaj in mesajlar]
        konu_sayısı = 3
        kmeans = KMeans(n_clusters=konu_sayısı, random_state=0)
        kmeans.fit(gömmeler)
        konu_etiketleri = kmeans.labels_

        for i, mesaj in enumerate(mesajlar):
            kullanıcı_profilleri[kullanıcı_kimliği]["ilgiler"].append({"mesaj": mesaj, "gömme": gömmeler[i], "konu": konu_etiketleri[i]})
        kullanıcı_profillerini_kaydet()


async def yeni_konu_öner(kullanıcı_kimliği):
    if kullanıcı_profilleri[kullanıcı_kimliği]["ilgiler"]:
        ilgiler = kullanıcı_profilleri[kullanıcı_kimliği]["ilgiler"]
        konu_sayıları = defaultdict(int)
        for ilgi in ilgiler:
            konu_sayıları[ilgi["konu"]] += 1
        en_sık_konu = max(konu_sayıları, key=konu_sayıları.get)
        önerilen_ilgi = random.choice([ilgi for ilgi in ilgiler if ilgi["konu"] == en_sık_konu])
        return f"Hey, maybe we could talk more about '{önerilen_ilgi['mesaj']}'? I'd love to hear your thoughts."
    else:
        return "I'm not sure what to talk about next. What are you interested in?"


class DiyalogDurumuİzleyici:
    durumlar = {"karşılama": {"giriş_eylemi": "kullanıcıyı_karşıla"}, "genel_konuşma": {}, "hikaye_anlatma": {}, "soru_cevap": {}, "planlama": {"giriş_eylemi": "planlamaya_başla"}, "çıkış_durumu": {"giriş_eylemi": "çıkışı_işle"}, "hata": {"giriş_eylemi": "hatayı_yönet"}}

    def __init__(self):
        self.makine = Machine(model=self, states=list(self.durumlar.keys()), initial="karşılama")
        self.makine.add_transition("karşıla", "karşılama", "genel_konuşma", conditions=["kullanıcı_merhaba_diyor"])
        self.makine.add_transition("soru_sor", "*", "soru_cevap", conditions=["kullanıcı_soru_soruyor"])
        self.makine.add_transition("hikaye_anlat", "*", "hikaye_anlatma", conditions=["kullanıcı_hikaye_istiyor"])
        self.makine.add_transition("planla", "*", "planlama", conditions=["kullanıcı_plan_istiyor"])
        self.makine.add_transition("çıkışı_işle", "*", "çıkış_durumu", conditions=["kullanıcı_çıkış_istiyor"])
        self.makine.add_transition("hata", "*", "hata")

    def kullanıcı_merhaba_diyor(self, kullanıcı_girdisi):
        return any(karşılama in kullanıcı_girdisi.lower() for karşılama in ["merhaba", "selam", "hey"])

    def kullanıcı_soru_soruyor(self, kullanıcı_girdisi):
        return any(soru_kelimesi in kullanıcı_girdisi.lower() for soru_kelimesi in ["ne", "kim", "nerede", "ne zaman", "nasıl", "neden"])

    def kullanıcı_hikaye_istiyor(self, kullanıcı_girdisi):
        return any(hikaye_anahtar_kelimesi in kullanıcı_girdisi.lower() for hikaye_anahtar_kelimesi in ["bana bir hikaye anlat", "bir hikaye anlat", "hikaye zamanı"])

    def kullanıcı_plan_istiyor(self, kullanıcı_girdisi):
        return any(plan_anahtar_kelimesi in kullanıcı_girdisi.lower() for plan_anahtar_kelimesi in ["bir plan yap", "bir şey planla", "planlamama yardım et"])

    def kullanıcı_çıkış_istiyor(self, kullanıcı_girdisi):
        return any(çıkış in kullanıcı_girdisi.lower() for çıkış in ["hoşçakal", "görüşürüz", "sonra görüşürüz", "çıkış"])


    def kullanıcıyı_karşıla(self, kullanıcı_kimliği):
        karşılamalar = [f"Merhaba <@{kullanıcı_kimliği}>! Bugün sana nasıl yardımcı olabilirim?", f"Selam <@{kullanıcı_kimliği}>, aklında ne var?", f"Hey <@{kullanıcı_kimliği}>! Senin için ne yapabilirim?"]
        return random.choice(karşılamalar)

    def planlamaya_başla(self, kullanıcı_kimliği):
        kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"] = {}
        return "Tamam, planlamaya başlayalım. Neyi planlamaya çalışıyorsun?"

    def çıkışı_işle(self, kullanıcı_kimliği):
        çıkışlar = [f"Hoşçakal, <@{kullanıcı_kimliği}>! İyi günler!", f"Görüşürüz, <@{kullanıcı_kimliği}>!", f"Sonra konuşuruz, <@{kullanıcı_kimliği}>!", f"Çıkış yapılıyor, <@{kullanıcı_kimliği}>!"]
        return random.choice(çıkışlar)

    def hatayı_yönet(self, kullanıcı_kimliği):
        return "Anlamadım. Lütfen isteğinizi yeniden ifade eder misiniz?"


    async def diyalog_eylemini_sınıflandır(self, kullanıcı_girdisi):
        for deneme in range(3):
            try:
                istem = f"Classify the following user input into one of the dialogue actions: karşılama, soru_cevap, hikaye_anlatma, genel_konuşma, planlama, çıkış. User input: {kullanıcı_girdisi} Classify the dialogue action by stating your answer as a single word on the first line:"
                yanıt = await groq_ile_yanıt_oluştur(istem)
                diyalog_eylemi = yanıt.strip().split("\n")[0].lower()
                return diyalog_eylemi

            except Exception as e:
                logging.error(f"Error occurred while extracting dialogue action: {e}, Attempt: {deneme + 1}")
                await asyncio.sleep(2**deneme)
                continue

        self.makine.trigger("hata")
        return self.makine.state



    async def durumu_geçiş_yap(self, geçerli_durum, kullanıcı_girdisi, kullanıcı_kimliği, konuşma_geçmişi):
        if self.makine.trigger("karşıla", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger("soru_sor", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger("hikaye_anlat", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger("planla", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        if self.makine.trigger("çıkışı_işle", kullanıcı_girdisi=kullanıcı_girdisi):
            return self.makine.state
        return "genel_konuşma"


diyalog_durumu_izleyici = DiyalogDurumuİzleyici()

ORAN_SINIRI_DAKİKADA_GEMINI = 60
ORAN_SINIRI_PENCERESİ_GEMINI = 60
kullanıcı_son_istek_zamanı_gemini = defaultdict(lambda: 0)
global global_son_istek_zamanı_gemini, global_istek_sayısı_gemini
global_son_istek_zamanı_gemini = 0
global_istek_sayısı_gemini = 0

def split_message(message: str, chunk_size: int = 2000) -> List[str]:
    chunks = []
    for i in range(0, len(message), chunk_size):
        chunks.append(message[i:i + chunk_size])
    return chunks


@backoff.on_exception(backoff.expo, (aiohttp.ClientError, Exception), max_tries=-1)
async def groq_ile_yanıt_oluştur(istem, kullanıcı_kimliği=None, dil="en"):
    client = Groq(api_key=groq_api_key)
    try:
        completion = await asyncio.to_thread(client.chat.completions.create, model="llama-3.2-90b-text-preview", messages=[{"role": "user", "content": istem}])
        groq_response = completion.choices[0].message.content
        cleaned_groq_response = re.sub(r'\(.*?\)', '', groq_response)
        cleaned_groq_response = re.sub(r"\[(.*?)\]", r"\1", cleaned_groq_response)
        return cleaned_groq_response
    except aiohttp.ClientError as e:
        logging.error(f"Groq API connection error: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding Groq response: {e}")
        raise
    except Exception as e:
        logging.exception(f"Unexpected error in Groq API call: {e}")
        raise


async def gemini_arama_ve_özetleme(sorgu):
    search_keywords = sorgu.split()[:5]
    search_query = " ".join(search_keywords)

    try:
        ddg = AsyncDDGS()
        arama_sonuçları = await asyncio.to_thread(ddg.text, search_query, max_results=50)

        if arama_sonuçları:
            arama_sonuçları_metni = ""
            for indeks, sonuç in enumerate(arama_sonuçları):
                arama_sonuçları_metni += f'[{indeks}] Başlık: {sonuç["title"]}\nÖzet: {sonuç["body"]}\nURL: {sonuç["href"]}\n\n'

            istem = f"You are a helpful AI assistant. A user asked about '{sorgu}'. Here are some relevant web search results:\n\n{arama_sonuçları_metni}\n\nPlease provide a concise and informative summary of these search results, prioritizing information relevant to the user's original query.  Cite the URLs used in your summary using [number] notation corresponding to the search results above.  If the query is about a website, provide only the URL of the website in the response."
            try:
                yanıt = await groq_ile_yanıt_oluştur(istem)
                logging.info(f"Raw Groq response: {yanıt}")
                return temizle_ve_url_al(yanıt)
            except Exception as e:
                logging.error(f"Groq API Error: {e}")
                return None
        else:
            return "I couldn't find any information about that on the web."

    except Exception as e:
        logging.error(f"Error during search and summarization: {e}")
        return None

def temizle_ve_url_al(text):
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\2", text)
    url_match = re.search(r"(https?://\S+)", text)
    if url_match:
        return url_match.group(1)
    else:
        return text

async def açıklamadan_url_çıkar(açıklama):
    arama_sorgusu = f"{açıklama} site:youtube.com OR site:twitch.tv OR site:instagram.com OR site:twitter.com"
    try:
        ddg = AsyncDDGS()
        results = await ddg.json(arama_sorgusu, max_results=1)
        if results and results['results']:
            return results['results'][0]['href']
        else:
            return None
    except Exception as e:
        logging.error(f"Error in açıklamadan_url_çıkar: {e}")
        return None


async def url_temizle(url, açıklama=None):
    if url is None:
        return None

    temizlenmiş_url = url.lower().strip()

    if not temizlenmiş_url.startswith(("https://", "http://")):
        temizlenmiş_url = "https://" + temizlenmiş_url

    temizlenmiş_url = re.sub(r"[^a-zA-Z0-9./?=-]", "", temizlenmiş_url)

    if "youtube.com" in temizlenmiş_url and "www.youtube.com" not in temizlenmiş_url:
        temizlenmiş_url = re.sub(r"(youtube\.com/)(.*)", r"www.youtube.com/\2", temizlenmiş_url)
    elif "twitch.tv" in temizlenmiş_url and "www.twitch.tv" not in temizlenmiş_url:
        temizlenmiş_url = re.sub(r"(twitch\.tv/)(.*)", r"www.twitch.tv/\2", temizlenmiş_url)
    elif "instagram.com" in temizlenmiş_url and "www.instagram.com" not in temizlenmiş_url:
        temizlenmiş_url = re.sub(r"(instagram\.com/)(.*)", r"www.instagram.com/\2", temizlenmiş_url)
    elif "twitter.com" in temizlenmiş_url and "www.twitter.com" not in temizlenmiş_url:
        temizlenmiş_url = re.sub(r"(twitter\.com/)(.*)", r"www.twitter.com/\2", temizlenmiş_url)



    try:
        yanıt = requests.get(temizlenmiş_url)
        if yanıt.status_code == 200:
            return temizlenmiş_url
        else:
            logging.warning(f"Cleaned URL ({temizlenmiş_url}) is not valid (status code: {yanıt.status_code}).")
            return None
    except requests.exceptions.RequestException:
        logging.warning(f"Cleaned URL ({temizlenmiş_url}) is not valid.")
        return None

async def karmaşık_diyalog_yöneticisi(kullanıcı_profilleri, kullanıcı_kimliği, mesaj):
    """Manages a complex dialogue for planning and execution."""

    profil = kullanıcı_profilleri.get(kullanıcı_kimliği)
    if not profil or profil["diyalog_durumu"] != "planlama":
        return "Dialogue is not in planning mode."  # Handle invalid state

    planlama_durumu = profil.setdefault("planlama_durumu", {})

    # Use a match-case statement for better readability and maintainability
    match planlama_durumu.get("aşama"):
        case "ilk_istek":
            hedef, sorgu_türü = await hedefi_çıkar(profil["sorgu"])
            planlama_durumu["hedef"] = hedef
            planlama_durumu["sorgu_türü"] = sorgu_türü
            planlama_durumu["aşama"] = "bilgi_toplama"
            return await açıklayıcı_sorular_sor(hedef, sorgu_türü)

        case "bilgi_toplama":
            await planlama_bilgisini_işle(kullanıcı_kimliği, mesaj)
            if await yeterli_planlama_bilgisi_var_mı(kullanıcı_kimliği):
                planlama_durumu["aşama"] = "plan_oluşturma"
                plan = await plan_oluştur(planlama_durumu["hedef"], planlama_durumu.get("tercihler", {}), kullanıcı_kimliği, mesaj)
                geçerli_mi, doğrulama_sonucu = await planı_doğrula(plan, kullanıcı_kimliği)
                if geçerli_mi:
                    planlama_durumu["plan"] = plan
                    planlama_durumu["aşama"] = "planı_sunma"
                    return await planı_sun_ve_geri_bildirim_iste(plan)
                else:
                    return f"Planlamada sorun var: {doğrulama_sonucu}. Daha fazla bilgi verin veya tercihlerinizi ayarlayın."
            else:
                return await daha_fazla_açıklayıcı_soru_sor(kullanıcı_kimliği)

        case "planı_sunma":
            geri_bildirim_sonucu = await plan_geri_bildirimini_işle(kullanıcı_kimliği, mesaj)
            if geri_bildirim_sonucu == "accept":
                planlama_durumu["aşama"] = "planı_değerlendirme"
                değerlendirme = await planı_değerlendir(planlama_durumu["plan"], kullanıcı_kimliği)
                planlama_durumu["değerlendirme"] = değerlendirme
                planlama_durumu["aşama"] = "planı_yürütme"
                ilk_yürütme_mesajı = await plan_adımını_yürüt(planlama_durumu["plan"], 0, kullanıcı_kimliği, mesaj)
                return await yanıt_oluştur(planlama_durumu["plan"], değerlendirme, {}, planlama_durumu.get("tercihler", {})) + "\n\n" + ilk_yürütme_mesajı
            else:
                planlama_durumu["aşama"] = "bilgi_toplama"
                return f"Tamam, planı gözden geçirelim. İşte bazı öneriler: {geri_bildirim_sonucu}. Hangi değişiklikleri yapmak istersiniz?"

        case "planı_yürütme":
            yürütme_sonucu = await plan_yürütmesini_izle(planlama_durumu["plan"], kullanıcı_kimliği, mesaj)
            return yürütme_sonucu

        case _:
            return "Geçersiz planlama aşaması." # Handle unexpected stage

async def açıklayıcı_sorular_sor(hedef, sorgu_türü):
    return "I need some more details to create an effective plan. Could you please tell me:\n- What is the desired outcome of this plan?\n- What are the key steps or milestones involved?\n- Are there any constraints or limitations I should be aware of?\n- What resources or tools are available?\n- What is the timeline for completing this plan?"


async def planlama_bilgisini_işle(kullanıcı_kimliği, mesaj):
    kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"]["kullanıcı_girdisi"] = mesaj.content


async def yeterli_planlama_bilgisi_var_mı(kullanıcı_kimliği):
    return "kullanıcı_girdisi" in kullanıcı_profilleri[kullanıcı_kimliği]["planlama_durumu"]["tercihler"]


async def daha_fazla_açıklayıcı_soru_sor(kullanıcı_kimliği):
    return "Please provide more details to help me create a better plan. For example, more information about steps, constraints, resources, or the time frame."


async def planı_sun_ve_geri_bildirim_iste(plan):
    plan_metni = ""
    for i, adım in enumerate(plan["adımlar"]):
        plan_metni += f"{i + 1}. {adım['açıklama']}\n"
    return f"Based on your input, a draft plan looks like this:\n\n{plan_metni}\n\nWhat do you think? Are there any changes you would like to make? (Type 'accept' to proceed)"


async def yanıt_oluştur(plan, değerlendirme, ek_bilgi, tercihler):
    yanıt = f"I've created a plan for your goal: {plan['hedef']}\n\n"

    yanıt += "**Steps:**\n"
    for i, adım in enumerate(plan["adımlar"]):
        yanıt += f"{i + 1}. {adım['açıklama']}"
        if "son_tarih" in adım:
            yanıt += f" (Deadline: {adım['son_tarih']})"
        yanıt += "\n"

    if değerlendirme:
        yanıt += f"\n**Evaluation:**\n{değerlendirme.get('değerlendirme_metni', '')}\n"

    if ek_bilgi:
        yanıt += "\n**Additional Information:**\n"
        for bilgi_türü, bilgi in ek_bilgi.items():
            yanıt += f"- {bilgi_türü}: {bilgi}\n"

    if tercihler:
        yanıt += "\n**Your Preferences:**\n"
        for tercih_adı, tercih_değeri in tercihler.items():
            yanıt += f"- {tercih_adı}: {tercih_değeri}\n"

    return yanıt


async def hedefi_çıkar(sorgu):
    istem = f"You are an AI assistant capable of understanding user goals. What is the user trying to achieve with the following query? User Query: {sorgu} Please specify the goal in a concise sentence."
    try:
        hedef = await groq_ile_yanıt_oluştur(istem)
    except Exception as e:
        logging.error(f"Error occurred while extracting user goal: {e}")
        return "I couldn't understand your goal. Please express it differently.", "general"
    return hedef.strip(), "general"


async def çok_aşamalı_duygu_analizi(sorgu, kullanıcı_kimliği):
    sonuçlar = []

    try:
        blob = TextBlob(sorgu)
        textblob_sentiment = blob.sentiment.polarity
        sonuçlar.append(textblob_sentiment)
    except Exception as e:
        logging.error(f"Error in TextBlob sentiment analysis: {str(e)}")


    try:
        vader = SentimentIntensityAnalyzer()
        vader_sentiment = vader.polarity_scores(sorgu)["compound"]
        sonuçlar.append(vader_sentiment)
    except Exception as e:
        logging.error(f"Error in VADER sentiment analysis: {str(e)}")

    try:
        sentiment_pipeline = pipeline("sentiment-analysis")
        transformer_sentiment = sentiment_pipeline(sorgu)[0]
        sonuçlar.append(transformer_sentiment["score"] if transformer_sentiment["label"] == "POSITIVE" else -transformer_sentiment["score"])
    except ImportError:
        logging.warning("Transformers library not found. Transformer-based sentiment analysis skipped.")
    except Exception as e:
        logging.error(f"Error in transformer sentiment analysis: {str(e)}")

    try:
        duygu_istemi = f"Analyze the sentiment and intensity of: {sorgu}. Return only the sentiment value as a float between -1 and 1."
        gemini_sentiment = await groq_ile_yanıt_oluştur(duygu_istemi, kullanıcı_kimliği)
        sentiment_match = re.search(r"-?\d+(\.\d+)?", gemini_sentiment)
        if sentiment_match:
            gemini_score = float(sentiment_match.group())
            sonuçlar.append(gemini_score)
        else:
            logging.error(f"Unable to extract sentiment value from Gemini response: {gemini_sentiment}")
    except Exception as e:
        logging.error(f"Error in Gemini sentiment analysis: {str(e)}")

    if sonuçlar:
        ortalama_duygu = np.mean(sonuçlar)
    else:
        logging.error("No valid sentiment scores obtained")
        ortalama_duygu = 0.0


    return {"duygu_etiketi": "olumlu" if ortalama_duygu > 0.05 else "olumsuz" if ortalama_duygu < -0.05 else "nötr", "duygu_yoğunluğu": abs(ortalama_duygu)}


def error_tracker(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.name}: {str(e)}", exc_info=True)
            raise

    return wrapper


veritabanı_kuyruğu = asyncio.Queue()


async def sohbet_geçmişini_kaydet(kullanıcı_kimliği, mesaj, kullanıcı_adı, bot_kimliği, bot_adı):
    await veritabanı_kuyruğu.put((kullanıcı_kimliği, mesaj, kullanıcı_adı, bot_kimliği, bot_adı))


async def veritabanı_kuyruğunu_işle():
    while True:
        while not veritabanı_hazır:
            await asyncio.sleep(1)
        kullanıcı_kimliği, mesaj, kullanıcı_adı, bot_kimliği, bot_adı = await veritabanı_kuyruğu.get()
        try:
            async with veritabanı_kilidi:
                async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
                    await db.execute("INSERT INTO sohbet_gecmisi (kullanıcı_kimliği, mesaj, zaman_damgası, kullanıcı_adı, bot_kimliği, bot_adı) VALUES (?, ?, ?, ?, ?, ?)", (kullanıcı_kimliği, mesaj, datetime.now(timezone.utc).isoformat(), kullanıcı_adı, bot_kimliği, bot_adı))
                    await db.commit()
        except Exception as e:
            logging.error(f"Error occurred while saving to the database: {e}")
        finally:
            veritabanı_kuyruğu.task_done()



async def geri_bildirimi_veritabanına_kaydet(kullanıcı_kimliği, geri_bildirim):
    async with veritabanı_kilidi:
        async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
            await db.execute("INSERT INTO geri_bildirimler (kullanıcı_kimliği, geri_bildirim, zaman_damgası) VALUES (?, ?, ?)", (kullanıcı_kimliği, geri_bildirim, datetime.now(timezone.utc).isoformat()))
            await db.commit()




async def ilgili_geçmişi_al(kullanıcı_kimliği, geçerli_mesaj):

    async with veritabanı_kilidi:
        geçmiş_metni = ""
        mesajlar = []
        async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
            async with db.execute("SELECT mesaj FROM sohbet_gecmisi WHERE kullanıcı_kimliği = ? ORDER BY id DESC LIMIT ?", (kullanıcı_kimliği, 50)) as cursor:
                async for satır in cursor:
                    mesajlar.append(satır[0])

        mesajlar.reverse()
        if not mesajlar:
            return ""


        tfidf_matrisi = tfidf_vektörleştirici.fit_transform(mesajlar + [geçerli_mesaj])
        geçerli_mesaj_vektörü = tfidf_matrisi[-1]
        benzerlikler = cosine_similarity(geçerli_mesaj_vektörü, tfidf_matrisi[:-1]).flatten()
        en_benzer_indeksler = np.argsort(benzerlikler)[-3:]

        for indeks in en_benzer_indeksler:
            geçmiş_metni += mesajlar[indeks] + "\n"
        return geçmiş_metni




async def sohbet_geçmişi_tablosu_oluştur():
    async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
        await db.execute("""CREATE TABLE IF NOT EXISTS sohbet_gecmisi (id INTEGER PRIMARY KEY, kullanıcı_kimliği TEXT, mesaj TEXT, zaman_damgası TEXT, kullanıcı_adı TEXT, bot_kimliği TEXT, bot_adı TEXT)""")
        await db.execute("""CREATE TABLE IF NOT EXISTS geri_bildirimler (id INTEGER PRIMARY KEY, kullanıcı_kimliği TEXT, geri_bildirim TEXT, zaman_damgası TEXT)""")
        await db.commit()


async def veritabanını_başlat():
    global veritabanı_hazır
    async with veritabanı_kilidi:
        await sohbet_geçmişi_tablosu_oluştur()
        veritabanı_hazır = True


def kullanıcı_profillerini_yükle():
    try:
        with open(KULLANICI_PROFİLLERİ_DOSYASI, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logging.warning("User profile file not found or corrupted. Starting new.")
        return defaultdict(lambda: {"tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []}, "demografi": {"yaş": None, "konum": None}, "geçmiş_özeti": "", "bağlam": [], "kişilik": {"mizah": 0.5, "nezaket": 0.8, "iddialılık": 0.6}, "diyalog_durumu": "karşılama", "uzun_süreli_hafıza": [], "son_bot_eylemi": None, "ilgiler": [], "sorgu": "", "planlama_durumu": {}, "etkileşim_geçmişi": [], "feedback_topics": [], "feedback_keywords": [], "satisfaction": 0})





def kullanıcı_profillerini_kaydet():
    profiller_kopyası = defaultdict(lambda: {"tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []}, "demografi": {"yaş": None, "konum": None}, "geçmiş_özeti": "", "bağlam": [], "kişilik": {"mizah": 0.5, "nezaket": 0.8, "iddialılık": 0.6}, "diyalog_durumu": "karşılama", "uzun_süreli_hafıza": [], "son_bot_eylemi": None, "ilgiler": [], "sorgu": "", "planlama_durumu": {}, "etkileşim_geçmişi": [], "feedback_topics": [], "feedback_keywords": [], "satisfaction": 0})

    for kullanıcı_kimliği, profil in kullanıcı_profilleri.items():
        profiller_kopyası[kullanıcı_kimliği].update(profil)
        profiller_kopyası[kullanıcı_kimliği]["bağlam"] = list(profil["bağlam"])

        for ilgi in profiller_kopyası[kullanıcı_kimliği]["ilgiler"]:
            if isinstance(ilgi.get("gömme"), np.ndarray):
                ilgi["gömme"] = ilgi["gömme"].tolist()

    try:
        with open(KULLANICI_PROFİLLERİ_DOSYASI, "w", encoding="utf-8") as f:
            json.dump(profiller_kopyası, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error occurred while saving user profiles: {e}")





async def veritabanından_geri_bildirimi_analiz_et():
    async with veritabanı_kilidi:
        async with aiosqlite.connect(VERİTABANI_DOSYASI) as db:
            async with db.execute("SELECT * FROM geri_bildirimler") as cursor:
                async for satır in cursor:
                    kullanıcı_kimliği, geri_bildirim, zaman_damgası = satır

                    duygu_istemi = f"Analyze the sentiment of the following feedback: Feedback: {geri_bildirim} Indicate the sentiment as one of the following: positive, negative, or neutral."
                    try:
                        duygu_yanıtı = await groq_ile_yanıt_oluştur(duygu_istemi, kullanıcı_kimliği)
                        duygu_etiketi = duygu_yanıtı.strip().lower()
                        logging.info(f"Sentiment Analysis of Feedback (User {kullanıcı_kimliği}): {duygu_etiketi}")
                    except Exception as e:
                        logging.error(f"Error occurred during sentiment analysis of feedback: {e}")
                        duygu_etiketi = "neutral"


                    try:
                        işlenmiş_geri_bildirim = geri_bildirimi_ön_işle(geri_bildirim)
                        tfidf = TfidfVectorizer().fit_transform([işlenmiş_geri_bildirim])
                        lda = LatentDirichletAllocation(n_components=3, random_state=0)
                        lda.fit(tfidf)
                        dominant_topic = np.argmax(lda.transform(tfidf))
                        logging.info(f"Dominant Topic for Feedback (User {kullanıcı_kimliği}): {dominant_topic}")

                        top_keywords = get_top_keywords_for_topic(lda, TfidfVectorizer().get_feature_names_out(), 5)
                        logging.info(f"Top Keywords for Topic {dominant_topic}: {top_keywords}")

                        if "feedback_topics" not in kullanıcı_profilleri[kullanıcı_kimliği]:
                            kullanıcı_profilleri[kullanıcı_kimliği]["feedback_topics"] = []
                        if "feedback_keywords" not in kullanıcı_profilleri[kullanıcı_kimliği]:
                            kullanıcı_profilleri[kullanıcı_kimliği]["feedback_keywords"] = []
                        kullanıcı_profilleri[kullanıcı_kimliği]["feedback_topics"].append(dominant_topic)
                        kullanıcı_profilleri[kullanıcı_kimliği]["feedback_keywords"].extend(top_keywords)
                    except Exception as e:
                        logging.error(f"Error occurred during topic modeling: {e}")


                    if duygu_etiketi == "positive":
                        kullanıcı_profilleri[kullanıcı_kimliği]["satisfaction"] = kullanıcı_profilleri[kullanıcı_kimliği].get("satisfaction", 0) + 1
                    elif duygu_etiketi == "negative":
                        logging.warning(f"Negative feedback received from User {kullanıcı_kimliği}: {geri_bildirim}")





def geri_bildirimi_ön_işle(geri_bildirim):
    return geri_bildirim



def get_top_keywords_for_topic(model, feature_names, num_top_words):
    topic_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        top_keywords_idx = topic.argsort()[:-num_top_words - 1:-1]
        topic_keywords.append([feature_names[i] for i in top_keywords_idx])
    return topic_keywords[topic_idx]


async def dil_tespit_et(metin):
    try:
        dil = detect(metin)
        return dil
    except LangDetectException:
        return "en"


async def çok_gelişmiş_muhakeme_gerçekleştir(içerik, ilgili_geçmiş, özetlenmiş_arama, kullanıcı_kimliği, mesaj, sorgu):

    try:
        dil = await dil_tespit_et(içerik)
        logging.info(f"Tespit edilen dil: {dil}")

        istem = f"Sen, karmaşık muhakeme yapabilen ve kullanıcı ihtiyaçlarını anlayabilen son derece gelişmiş bir AI asistansısın. Kullanıcının mesajı: {içerik} İlgili sohbet geçmişi: {ilgili_geçmiş} Özetlenmiş arama sonuçları: {özetlenmiş_arama if özetlenmiş_arama else 'Arama sonucu bulunamadı.'} Kullanıcının sorgusuna ayrıntılı ve yararlı bir yanıt üretin. Yanıtın özel, ilgili ve kullanıcının ihtiyaçlarını karşıladığından emin olun."

        groq_yanıtı = await groq_ile_yanıt_oluştur(istem, kullanıcı_kimliği, dil)

        birleşik_yanıt = f"**Groq:** {groq_yanıtı}"


        return birleşik_yanıt, "Groq"

    except Exception as e:
        logging.error(f"Çok gelişmiş muhakeme fonksiyonunda hata: {str(e)}")
        return None, None





async def ilgili_geçmişi_al(kullanıcı_kimliği, içerik):
    try:
        geçmiş = kullanıcı_profilleri[kullanıcı_kimliği]["bağlam"]
        ilgili_geçmiş = "\n".join([f"Kullanıcı: {item['içerik']}" for item in geçmiş])
        return ilgili_geçmiş
    except Exception as e:
        logging.error(f"İlgili geçmişi alma fonksiyonunda hata: {str(e)}")
        return ""



async def kullanıcı_ilgi_alanlarını_belirle(kullanıcı_kimliği, içerik):
    if "music" in içerik:
        kullanıcı_profilleri[kullanıcı_kimliği]["ilgiler"].append("music")
    elif "movies" in içerik:
        kullanıcı_profilleri[kullanıcı_kimliği]["ilgiler"].append("movies")




@bot.event
async def on_ready():
    logging.info(f"{bot.user} olarak giriş yapıldı!")
    bot.loop.create_task(veritabanı_kuyruğunu_işle())
    change_status.start()


@tasks.loop(minutes=5)
async def change_status():
    statuses = [
        "Dünyayı keşfediyor...",
        "Yeni şeyler öğreniyor...",
        "İnsanlarla sohbet ediyor...",
        "Hayatı sorguluyor...",
        "Evreni anlamaya çalışıyor...",
        "Kod yazıyor...",
        "Şiir okuyor...",
        "Müzik dinliyor...",
        "Film izliyor...",
        "Rüya görüyor...",
    ]
    await bot.change_presence(activity=discord.Game(random.choice(statuses)))



@bot.event
async def on_message(mesaj):
    global aktif_kullanıcılar, hata_sayacı, yanıt_süresi_histogramı, yanıt_süresi_özeti
    if mesaj.author == bot.user or bot.user.mentioned_in(mesaj) is False:
        return


    aktif_kullanıcılar += 1
    kullanıcı_kimliği = str(mesaj.author.id)
    içerik = mesaj.content.strip()

    try:
        if kullanıcı_kimliği not in kullanıcı_profilleri:
            kullanıcı_profilleri[kullanıcı_kimliği] = {
                "tercihler": {"iletişim_tarzı": "samimi", "ilgi_alanları": []},
                "demografi": {"yaş": None, "konum": None},
                "geçmiş_özeti": "",
                "bağlam": deque(maxlen=BAĞLAM_PENCERESİ_BOYUTU),
                "kişilik": {"mizah": 0.5, "nezaket": 0.8, "iddialılık": 0.6, "yaratıcılık": 0.5},
                "diyalog_durumu": "karşılama",
                "uzun_süreli_hafıza": [],
                "son_bot_eylemi": None,
                "ilgiler": [],
                "sorgu": "",
                "planlama_durumu": {},
                "etkileşim_geçmişi": [],
                "feedback_topics": [],
                "feedback_keywords": [],
                "satisfaction": 0,
                "duygusal_durum": "nötr",
                "çıkarımlar": [],
            }
        elif not isinstance(kullanıcı_profilleri[kullanıcı_kimliği].get("bağlam"), deque):
            kullanıcı_profilleri[kullanıcı_kimliği]["bağlam"] = deque(kullanıcı_profilleri[kullanıcı_kimliği].get("bağlam", []), maxlen=BAĞLAM_PENCERESİ_BOYUTU)


        kullanıcı_profilleri[kullanıcı_kimliği]["bağlam"].append({"rol": "kullanıcı", "içerik": içerik})
        kullanıcı_profilleri[kullanıcı_kimliği]["sorgu"] = içerik


        await kullanıcı_ilgi_alanlarını_belirle(kullanıcı_kimliği, içerik)


        ilgili_geçmiş = await ilgili_geçmişi_al(kullanıcı_kimliği, içerik)
        özetlenmiş_arama = await gemini_arama_ve_özetleme(içerik)

        başlangıç_zamanı = time.time()
        async with mesaj.channel.typing():
            try:
                yanıt_metni, kaynak = await çok_gelişmiş_muhakeme_gerçekleştir(içerik, ilgili_geçmiş, özetlenmiş_arama, kullanıcı_kimliği, mesaj, içerik)
                if yanıt_metni:
                    for chunk in split_message(yanıt_metni):
                        await mesaj.channel.send(chunk)
                else:
                    await mesaj.channel.send("Üzgünüm, şu anda bir yanıt oluşturamiyorum.")
            except asyncio.TimeoutError:
                await mesaj.channel.send("Yanıt oluşturma zaman aşımına uğradı.")
            except Exception as e:
                logging.exception(f"Error during LLM processing: {e}")
                await mesaj.channel.send("Bir hata oluştu. Lütfen daha sonra tekrar deneyin.")


        bitiş_zamanı = time.time()
        yanıt_süresi = bitiş_zamanı - başlangıç_zamanı
        yanıt_süresi_histogramı.append(yanıt_süresi)
        yanıt_süresi_özeti = f"Yanıt süresi: {yanıt_süresi:.2f} saniye"

        await sohbet_geçmişini_kaydet(kullanıcı_kimliği, mesaj.content, mesaj.author.name, bot.user.id, bot.user.name)

    except Exception as e:
        logging.error(f"on_message fonksiyonunda hata oluştu: {str(e)}", exc_info=True)
        hata_sayacı += 1


    finally:
        aktif_kullanıcılar -= 1

async def bot_setup():
    bot.loop.create_task(veritabanını_başlat())

bot.setup_hook = bot_setup
bot.run(discord_token)
