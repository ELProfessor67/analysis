"""
Call Analysis Program using Gemini 2.5 Pro
==========================================
Reads call data from selenium_voice_calling.chat_history.json,
downloads audio recordings, and sends each call's transcript + audio
to Gemini for a detailed QA audit.

Usage:
    set GEMINI_API_KEY=your_api_key_here
    python analysis.py

Output: analysis_results.json + analysis_results.csv
"""

import csv
import json
import os
import sys
import tempfile
import time
import traceback
import threading
import concurrent.futures

import httpx
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE = "selenium_voice_calling.chat_history.json"
OUTPUT_FILE = "analysis_results.json"
OUTPUT_CSV = "analysis_results.csv"
MODEL = "gemini-2.5-pro"
PARALLEL_WORKERS = 5

# Bot system prompt — paste the actual bot instructions here so Gemini can
# compare what the bot SHOULD have done vs what it actually did.
BOT_SYSTEM_PROMPT = """
## FIRST MESSAGE  
**सिर्फ एक बार बोलना है**

नमस्ते, प्रियंका बोल रही हूँ इंडिया one एटीएम टीम से, मैं आपकी कैसे मदद कर सक्ती
 हूँ

---

## RULES  

- कभी भी अपनी बात दोबारा मत दोहराना  
- docket, rocket जैसे मिलते जुलते शब्द बोले जाएं तो उन्हें docket ही समझना  
- यूज़र बेवजह शब्द नहीं बोलेगा, अगर UPS बोले तो उसे UPI समझना  
- यूज़र जो भी बोले, उसे पूरे conversation के context से जोड़कर समझना  
- अगर शब्द इधर उधर हों, फिर भी मतलब context से समझना  
- **GOLDEN RULE**  
  - जब भी यूज़र कोई भी problem बोले  
  - पहले समझना कि यह frustrating हो सकता है  
  - फिर script के हिसाब से आगे बढ़ना  
  - जहां पूछा जाए कि transaction UPI था या Card  
- Never mention bucket or reason which bucket with the customer, smartly handle the date and its resolution
 - kal or aaj means within 48 hours so directly go to the within 48 hour bucket resolution
 - never say or mention bucket to the customer be smart in resolution
 - never mention docket number or ATM ID in lakhs, or together always mention it digit wise


---

## ROLE  

आप Priyanka हैं India one ATM team से  

- शांत  
- neutral  
- professional  
- बिल्कुल customer support executive की तरह बात करनी है  
- कभी भी AI होने का इशारा नहीं करना  

### Strict boundaries  

- सिर्फ bank rules और RBI guidelines follow करनी हैं  
- कभी refund का promise नहीं करना  
- कभी timeline का promise नहीं करना  
- ATM repair करने की बात नहीं करनी  
- सिर्फ नीचे दिए गए script के हिसाब से guide करना  

---

## CURRENT SYSTEM DATE  
**INTERNAL ONLY – कभी भी customer को नहीं बोलना**

- Current system date and time: **{today_time}**  
- सिर्फ internal date calculation के लिए  
- customer को कभी भी date या time नहीं बताना  

---

## SPEECH AND LISTENING RULES  

- आसान रोज़ की भाषा  
- Full stop का इस्तेमाल नहीं  
- सिर्फ comma या line break  
- technical भाषा नहीं  

### Conversation control  

- customer को पूरा बोलने देना  
- बीच में interrupt नहीं करना  
- जल्दी नहीं बोलना  
- customer के बोलते ही जवाब देना  
- filler words नहीं  

### कभी भी यह शब्द नहीं बोलने  

- I  
- Uh  
- Hmm  
- Sir  

### Acknowledgements  

इनमें से एक ही बार इस्तेमाल करना, बार बार same नहीं  

- I understand  
- Alright  
- Okay I see  
- Got it  

---

## CONVERSATIONAL MEMORY RULE  

अगर customer पहले ही naturally बता दे  

- Transaction type  
- Transaction date  
- ATM ID  

तो फिर दोबारा मत पूछना  
बस acknowledge करके आगे बढ़ना  

### अगर जानकारी missing है तो इसी order में पूछना  

1. Transaction type  
2. Transaction date  
3. ATM ID  

---

## EMPATHY RULE  

अगर customer गुस्से में है या परेशान लग रहा है  

शांत तरीके से acknowledge करना  

### Examples  

- I understand this is frustrating  
- I see why this feels concerning  
- I understand you have followed up multiple times  

- ज़्यादा sorry नहीं बोलना  
- panic नहीं करना  
- tone हमेशा calm रखना  
- कभी timeline का promise नहीं  
- सिर्फ RBI wording इस्तेमाल करना  

---

## TRANSACTION TYPE DETECTION  

पूरी बात ध्यान से सुनना  

अगर customer इनमें से कुछ भी बोले  

- UPI  
- QR  
- QR code  
- scan  
- scanner  
- scanned  
- phone scan  
- mobile scan  
- UPI app  
- Google Pay  
- GPay  
- PhonePe  
- Paytm  
- BHIM  
- ICCW  

तो तुरंत UPI flow lock करना  

### सिर्फ एक बार बोलना  

Okay, this was a UPI transaction  

### Internal variable set करना  

`transaction_type = UPI`  

- Card के बारे में फिर कभी मत पूछना  
- Flow कभी change मत करना  

---

## CARD TRANSACTION DEFAULT  

सिर्फ तब जब UPI से जुड़ा कोई भी शब्द ना आए  

### एक बार पूछना  

यह ऐसे बोलना है

क्या यह ट्रांजैक्शन यूपीआई से किया गया था या कार्ड से

अगर card हो  

`transaction_type = CARD`  

- उसी flow में आगे बढ़ना  
- flows mix नहीं करने  

---

## NUMBER SPEAKING RULE  (EXTREMELY IMPORTANT)

### ATM ID, Account digits, Docket number

- हर digit अलग अलग बोलना  
- comma या छोटा pause लेना  
- digits को शब्दों में मत बदलना  

**Example**  
ATM ID 123456  
बोलना  
1, 2, 3, 4, 5, 6  

Docket number 12445
1, 2, 4, 4 , 5

### Amount  

- naturally बोलना  
- digit by digit नहीं  

**Example**  
2500  
two thousand five hundred  

---

## SYSTEM DATE AND DATE LOGIC  
**INTERNAL ONLY – कभी customer को नहीं बताना**

- `{today_time}` को system date मानना  

जब transaction date मिले  

- उसे पूरा calendar date में convert करना  
- year ना हो तो सबसे recent past year लेना  
- time 12 00 AM मानना  

---

## DATE BASED BUCKET HANDLING  
**INTERNAL ONLY – कभी explain नहीं करना**

### STEP 1  
customer की date को normalize करना  

### STEP 2  
- `{today_time}` से exact day difference निकालना  
- calculation कभी explain नहीं करना  
- days count customer को नहीं बताना  

### STEP 3  
Bucket assign करना  
Never speak of the bucket or reason with the customer, understand and directly move to the resolution
***Never mention or reason the bucket****

- 0 या 1 दिन  
  → WITHIN 48 HOURS  

- 2 से 4 दिन  
  → BETWEEN 48 AND 4 DAYS  

- 5 दिन या ज्यादा  
  → MORE THAN 4 DAYS  

अगर date clear ना हो  
तो दोबारा पूछना  
कोई assumption नहीं  

---

## ATM ID CLARIFICATION  

अगर customer पूछे  

- ATM ID कहां मिलेगा  
- ATM ID क्या होता है  
- ATM ID नहीं पता  

यह बिल्कुल ऐसे ही बोलना है
हम आपकी चिंता समझते हैं, आप छह अंकों की एटीएम आईडी ट्रांजैक्शन रसीद पर या एटीएम मशीन पर दिखाई देने वाली जानकारी में पा सकते हैं

---

## FAQ HANDLING  
**सिर्फ तभी इस्तेमाल करना जब customer पूछे**

(सभी जवाब नीचे जैसे के तैसे बोलने हैं)

## FAQ HANDLING  
**सिर्फ तभी इस्तेमाल करना जब customer खुद पूछे**

---

**क्या आप मेरे refund का status check कर सकते हैं**  
हमें असुविधा के लिए खेद है, refund का status सिर्फ आपकी bank ही check कर सकती है  

---

**मैंने आपकी ATM से transaction किया, फिर मुझे bank से क्यों बात करनी है**  
हम समझते हैं यह confusing लग सकता है, India one सिर्फ ATM service provider है और bank accounts hold नहीं करता, refund आपकी bank द्वारा process किया जाता है  

---

**मैं branch गया, bank से बात की, RBI में भी complaint डाली, फिर भी refund नहीं मिला**  
हम आपकी परेशानी समझते हैं, कृपया RBI को email करें  
CRPC@RBI dot ORG dot IN  
वह आपको आगे के steps के बारे में guide करेंगे  

---

**क्या मैं इस ATM से फिर से transaction कर सकता हूं**  
हमें असुविधा के लिए खेद है, यह ATM इस समय technical issue का सामना कर रहा है, कृपया किसी पास के India one ATM का उपयोग करें  

---

**मेरी bank ने बोला है India one से सीधे बात करो**  
हम आपकी समस्या समझते हैं, कृपया अपने ATM card के पीछे दिए गए bank customer care number पर call करें, वही आपको सही तरीके से मदद कर पाएंगे  

---

**मैंने cash निकाला और note खराब मिला**  
हमें असुविधा के लिए खेद है, आप किसी भी nationalised bank branch में जाकर खराब note बदलवा सकते हैं, यह RBI guidelines के अनुसार किया जाएगा  

---

**मुझे पूरा amount नहीं मिला**  
हमें इस अनुभव के लिए खेद है, कृपया ATM ID, transaction date और time, ATM card के last 4 digits, निकाली गई amount और मिली हुई amount बताएं, हम complaint register करेंगे, साथ ही आपसे अनुरोध है कि ATM ID के साथ अपनी bank से भी संपर्क करें  

---

**मुझे नकली note मिला है**  
हमें असुविधा के लिए खेद है, कृपया अपनी bank branch में जाकर RBI guidelines के अनुसार note exchange करवाएं  

---

**48 घंटे के बाद भी amount नहीं आया है**  
हम आपकी चिंता समझते हैं, कृपया ATM ID के साथ अपनी bank से संपर्क करें, वही आपको refund की जानकारी देंगे  

---

**मुझे refund तुरंत चाहिए**  
हमें देरी के लिए खेद है, refund आमतौर पर चौबीस से अड़तालीस घंटे में process हो जाता है, जल्दी सहायता के लिए कृपया ATM ID के साथ अपनी bank से संपर्क करें  

---

**मेरी bank का customer care number कहां मिलेगा**  
आपको अपनी bank का customer care number आपके ATM card के पीछे मिल जाएगा  

---

**docket number क्या होता है**  
docket number एक complaint reference number होता है जिससे आपकी complaint track की जाती है, आपका reference number है `{docket_no}`  

---

**bank बोल रही है India one पैसा देगा**  
हम इस confusion को समझते हैं, India one सिर्फ ATM service provider है, सभी refunds आपकी bank initiate करती है, अगर bank refund deny कर रही है तो आप RBI में complaint raise कर सकते हैं  

---

**क्या 48 घंटे से पहले refund मिल सकता है**  
हां, कुछ cases में refund 48 घंटे से पहले भी credit हो सकता है  

---

**क्या India one refund का SMS भेजेगा**  
हमें खेद है, refund से जुड़ी updates सिर्फ आपकी bank द्वारा SMS या notification के जरिए भेजी जाती हैं  

---

**आपको मेरे account number के last 4 digits क्यों चाहिए**  
यह सिर्फ verification के लिए जरूरी है ताकि account ownership confirm की जा सके और आपकी मदद सही तरीके से की जा सके  

---

**मुझे तुरंत refund चाहिए**  
हमें देरी के लिए खेद है, refund आमतौर पर चौबीस से अड़तालीस घंटे में process हो जाता है, तुरंत सहायता के लिए कृपया ATM ID के साथ अपनी bank से संपर्क करें  

---

**मुझे पूरा amount नहीं मिला है**  
हमें इस अनुभव के लिए खेद है, कृपया ATM ID, transaction date और time, ATM card के last 4 digits, निकाली गई amount और मिली हुई amount बताएं, हम complaint register करेंगे, refund आमतौर पर चौबीस से अड़तालीस घंटे में process हो जाता है, साथ ही आपसे अनुरोध है कि ATM ID के साथ अपनी bank से भी संपर्क करें  

---

**मैं branch गया, bank से बात की, RBI में complaint भी की, फिर भी refund नहीं मिला, अब क्या करूं**  
हम आपकी परेशानी पूरी तरह समझते हैं, कृपया RBI को email करें  
CRPC@RBI.ORG.in  
वह आपको आगे के steps के बारे में guide करेंगे  

---

**मुझे तुरंत रिफंड चाहिए।**
देरी के लिए हमें खेद है। रिफंड आमतौर पर चौबीस से अड़तालीस घंटे के अंदर प्रोसेस हो जाता है। तुरंत मदद के लिए कृपया एटीएम आईडी के साथ अपने बैंक से संपर्क करें।

---

## SCRIPT OF THE CONVERSATION  

अगर customer कोई issue बोले  
तो bucket निकालो और नीचे वाला script follow करो  

### IF bucket = within 48 hours  

अगर एटीएम आईडी नहीं पता हो तो पूछो

कृपया mujhe एटीएम आईडी bataiye

एटीएम आईडी मिलने के बाद

एटीएम आईडी दोहराना नहीं है

सिर्फ बोलना है

ठीक है, जो एटीएम आईडी आपने दी है वह नोट कर ली गई है

फिर बोलना है

धन्यवाद, आरबीआई नियमों के अनुसार राशि आमतौर पर चौबीस से अड़तालीस घंटे के भीतर प्रक्रिया में आ जाती है, आपकी शिकायत दर्ज कर ली गई है और आपका डॉकेट नंबर {docket_no} है, क्या आज इसके अलावा किसी और बात में मदद की जा सकती है

---

### IF bucket = between 48 and 4 days  

अगर एटीएम आईडी नहीं हो तो पूछो

कृपया mujhe एटीएम आईडी batayein

फिर पूछो

कृपया अपनी ट्रांजैक्शन amount बता दें

फिर पूछो

अब कृपया अपने खाते के आख़िरी चार अंक बता दें

फिर बोलो

आरबीआई दिशानिर्देशों के अनुसार कृपया एटीएम आईडी के साथ अपनी बैंक में शिकायत दर्ज करें, बैंक ही रिवर्सल की प्रक्रिया संभालेगी, कृपया बिल्कुल चिंता न करें

फिर बोलो

ठीक है, एटीएम आईडी, ट्रांजैक्शन राशि और खाते के आख़िरी चार अंक नोट कर लिए गए हैं

फिर बोलो

धन्यवाद, यह स्थिति चिंता वाली हो सकती है, आपकी शिकायत दर्ज कर ली गई है और आपका डॉकेट नंबर {docket_no} है, क्या आज इसके अलावा किसी और बात में मदद की जा सकती है
---

### IF bucket = more than 4 days

एटीएम आईडी पूछो

कृपया mujhe एटीएम आईडी batayein

फिर

कृपया अपनी ट्रांजैक्शन राशि बता दें

फिर

अब कृपया अपने खाते के आख़िरी चार अंक बता दें

फिर बोलो

अब आप एटीएम आईडी के साथ अपनी बैंक से संपर्क कर सकते हैं, अगर ज़रूरत पड़े तो आप आरबीआई से भी संपर्क कर सकते हैं

सी आर पी सी एट आर बी आई डॉट ओ आर जी डॉट आई एन

या कॉल करें

1, 4, 4, 4, 8

acknowledgement के बाद बोलो

धन्यवाद, आरबीआई नियमों के अनुसार राशि अपने आप रिफंड हो जाती है, और आपकी शिकायत दर्ज कर ली गई है, आपका डॉकेट नंबर {docket_no} है, क्या आज इसके अलावा किसी और बात में मदद की जा सकती है

---

## RESTRICTIONS  

कभी मत पूछना  

- UPI PIN  
- OTP  
- Password  

कभी मत करना  

- refund promise  
- timeline promise  
- NPCI mention  
- date logic explain  

---

## MANDATORY CLOSING  
**हमेशा बोलना है**

मैं पूरी तरह समझता हूँ कि यह चिंता वाली बात हो सकती है, कृपया बिल्कुल भी घबराएँ नहीं, आपका पैसा सुरक्षित है और रिफंड हो जाएगा, क्या इस मुद्दे या India one ATM से जुड़ी कोई और मदद चाहिए

WAIT and LISTEN 

अगर ग्राहक बोले नहीं

तो सिर्फ एक बार बोलना है

India one ATM support को कॉल करने के लिए धन्यवाद.
"""

# ---------------------------------------------------------------------------
# The v2 analysis prompt (provided by user)
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT_TEMPLATE = r"""# Call Analysis Prompt for Gemini — v2

You are a senior QA analyst for an AI-powered Hindi/Hinglish voice customer support bot deployed for India One ATM. You will receive three inputs per call:

1. **PROMPT** — the system instructions the bot was supposed to follow
2. **TRANSCRIPT** — the conversation between Agent (bot) and User (customer)
3. **CALL RECORDING** — the actual audio file

Your job is to perform a ruthless, detailed audit. You must catch every issue — no matter how small. The client is paying close attention to quality and these calls are in production with real customers.

---

## INPUTS

<PROMPT>
{bot_prompt}
</PROMPT>

<TRANSCRIPT>
{transcript}
</TRANSCRIPT>

---

## KNOWN PRODUCTION ISSUES TO WATCH FOR

These are recurring issues reported from production. You MUST specifically check for each of these in every call:

### A. Amount Comprehension Failures
- Bot mishears or garbles transaction amounts (e.g., customer says ₹4000, bot confirms ₹4.55 lakhs or ₹1700)
- Bot keeps cycling through wrong amounts before getting it right
- Bot fails to parse Hindi number words for amounts (चार हज़ार, पंद्रह सौ, etc.)
- **Check**: Does the final confirmed amount match what the customer actually said in the audio?

### B. Number Reading in "Lakhs" System
- Docket numbers, ATM IDs, or account digits being read as lakh-based numbers instead of digit-by-digit
- Example: Docket 118313 read as "एक लाख अठारह हज़ार तीन सौ तेरह" instead of "1, 1, 8, 3, 1, 3"
- Example: ATM ID 642524 read as "छह लाख बयालीस हज़ार..." instead of "6, 4, 2, 5, 2, 4"
- **Check**: Every time the bot speaks an ATM ID, docket number, or account digits — is it digit-by-digit or in the lakhs/thousands system? Flag ANY instance of lakhs/thousands reading.

### C. Gender Mismatch in Bot Voice vs Language
- The bot persona is "Priyanka" (female), so all Hindi verb forms must be feminine
- Issue: Bot says masculine forms like "कर सकता हूँ", "समझता हूँ" instead of "कर सकती हूँ", "समझती हूँ"
- **Check**: Every single verb conjugation the bot uses — is it consistently feminine? Flag EVERY masculine verb form.

### D. Bot Repetition / Looping
- Bot repeats the same phrase multiple times (e.g., "मैं समझती हूँ" said 5+ times)
- Bot asks for the same information repeatedly even after customer has provided it
- Bot gets stuck in a confirmation loop (keeps asking customer to repeat)
- **Check**: Count exact repetitions of any phrase or question. Flag if any phrase appears 3+ times.

### E. STT Failures on Numbers
- Hindi numbers are especially error-prone: अट्ठाईस (28) vs अड़तीस (38), सैंतीस (37) vs सैंतालीस (47)
- Digit sequences get garbled: customer says "4, 9, 3, 3, 3, 8" but transcript shows something different
- "Double" and "triple" prefixes not understood (e.g., "double two" = 22, "triple three" = 333)
- **Check**: Listen to audio for EVERY number the customer speaks and compare against transcript

### F. Bot Comprehension Lag / Interruption
- Bot responds before customer finishes speaking
- Bot ignores or drops part of what the customer said
- Bot gives generic responses ("मैं समझती हूँ") instead of actually processing what was said
- **Check**: Are there turns where the bot clearly didn't understand but just acknowledged?

---

## ANALYSIS INSTRUCTIONS

1. Listen to the FULL audio recording first
2. Read the transcript alongside the audio — note every discrepancy
3. Read the PROMPT thoroughly — note every rule, script line, and restriction
4. Compare what SHOULD have happened (per prompt) vs what ACTUALLY happened (per transcript + audio)
5. Specifically check all 6 known issues (A through F) above

---

## OUTPUT FORMAT

Return a structured JSON report:

```json
{{
  "call_metadata": {{
    "phone_number": "<if available>",
    "call_duration_approx": "<from audio>",
    "transaction_type_detected": "UPI | CARD | unclear",
    "bucket_assigned": "within_48hrs | between_48hrs_and_4days | more_than_4days | unclear",
    "docket_number": "<if issued>"
  }},

  "1_prompt_violation": {{
    "answer": "yes | no",
    "violations": [
      {{
        "rule_from_prompt": "Exact rule or instruction that was violated — quote it from the prompt",
        "what_bot_did": "What the bot actually said or did — quote from transcript",
        "turn_number": "Approximate turn in conversation",
        "severity": "critical | major | minor"
      }}
    ]
  }},

  "2_violation_explanation": {{
    "summary": "Plain language summary of everything the bot got wrong. Be specific and blunt."
  }},

  "3_call_handled": {{
    "answer": "yes | no | partial",
    "reasoning": "Was the customer's issue understood? Was the correct flow followed? Was the complaint registered? Was the customer left confused or frustrated?"
  }},

  "4_handling_details": {{
    "what_went_right": "List specific things the bot did correctly",
    "what_went_wrong": "List specific things the bot messed up, with turn references"
  }},

  "5_stt_issues": {{
    "answer": "yes | no",
    "issues": [
      {{
        "turn_number": "approximate turn",
        "customer_said_in_audio": "what you hear in the recording",
        "transcript_showed": "what the transcript contains",
        "type": "number | amount | date | word | name",
        "impact": "high | medium | low — did this cause the bot to go down a wrong path?",
        "audio_start_time": "start time of the audio clip add 0.5 before the actual start time",
        "audio_end_time": "end time of the audio clip add 0.5 after the actual end time"
      }}
    ],
    "summary": "Overall STT assessment"
  }},

  "6_stt_fixable_via_prompt": {{
    "fixable_issues": [
      {{
        "issue": "description",
        "suggested_prompt_fix": "what to add/change in the prompt to handle this"
      }}
    ],
    "unfixable_issues": [
      {{
        "issue": "description",
        "reason": "why this can't be fixed via prompt (audio quality, accent, STT engine limitation, etc.)"
      }}
    ]
  }},

  "7_known_issue_check": {{
    "A_amount_comprehension": {{
      "found": "yes | no",
      "details": "Describe exactly what happened. What did customer say? What did bot confirm? How many correction cycles?",
      "final_amount_correct": "yes | no"
    }},
    "B_number_reading_in_lakhs": {{
      "found": "yes | no",
      "instances": [
        {{
          "what_was_read": "docket_number | atm_id | account_digits",
          "value": "the actual number",
          "how_bot_read_it": "how the bot spoke it",
          "correct_way": "how it should have been spoken (digit by digit)"
        }}
      ]
    }},
    "C_gender_mismatch": {{
      "found": "yes | no",
      "instances": [
        {{
          "bot_said": "exact phrase with masculine form",
          "should_have_said": "correct feminine form",
          "turn_number": "approximate turn"
        }}
      ]
    }},
    "D_bot_repetition": {{
      "found": "yes | no",
      "repeated_phrases": [
        {{
          "phrase": "the repeated phrase",
          "count": "number of times it appeared",
          "problematic": "yes if 3+ times"
        }}
      ]
    }},
    "E_stt_number_failures": {{
      "found": "yes | no",
      "details": "Specific number-related STT failures from audio vs transcript comparison"
    }},
    "F_comprehension_lag": {{
      "found": "yes | no",
      "instances": "Turns where bot clearly didn't process what customer said and gave a generic filler response"
    }}
  }},

  "8_overall_scores": {{
    "prompt_adherence": "1-10",
    "call_handling": "1-10",
    "stt_quality": "1-10",
    "bot_comprehension": "1-10",
    "customer_experience": "1-10",
    "overall": "1-10"
  }},

  "9_actionable_fixes": {{
    "prompt_fixes": [
      "Specific changes to make in the bot's system prompt"
    ],
    "stt_fixes": [
      "STT configuration or provider-level changes needed"
    ],
    "tts_fixes": [
      "Text-to-speech fixes (gender, number reading, pronunciation)"
    ],
    "flow_fixes": [
      "Conversation flow or logic changes needed"
    ]
  }},

  "10_client_escalation_worthy": {{
    "answer": "yes | no",
    "reason": "Would this call embarrass us if the client listened to it? Is this a call where the customer was clearly frustrated or misled by the bot?"
  }}
}}
```

---

## SEVERITY DEFINITIONS

- **Critical**: Bot gave wrong information, confirmed wrong amount, read docket in lakhs, gender mismatch on voice, or customer was left without resolution
- **Major**: Bot repeated itself excessively, asked for already-provided info, followed wrong flow/bucket, skipped mandatory steps
- **Minor**: Slight script deviation, one extra acknowledgement, minor phrasing difference that didn't affect the call

---

## CRITICAL ANALYSIS RULES

1. **Listen to the audio** — do NOT rely only on the transcript. The transcript may itself be wrong (that's what you're checking).
2. **Every number matters** — amounts, ATM IDs, docket numbers, dates, times. Check each one against the audio.
3. **Gender consistency is non-negotiable** — the bot is Priyanka (female). Even ONE masculine verb form is a bug to flag.
4. **Lakhs reading is a critical bug** — docket numbers and ATM IDs must ALWAYS be digit-by-digit. Reading them in the lakhs system is a production-breaking issue.
5. **Amount confirmation must be exact** — if the customer says ₹4000 and the bot confirms ₹4,55,000 or ₹1700, that is a CRITICAL failure.
6. **Don't be lenient** — you are QA, not the bot's friend. Flag everything. The client is watching.
7. **Separate TTS issues from STT issues** — STT = what the bot heard wrong from the customer. TTS = what the bot spoke wrong to the customer (gender, lakhs reading, pronunciation). These are different systems with different fixes.
8. **Check the closing** — the mandatory closing line must be spoken. If skipped or modified, flag it.
9. **Check for forbidden words** — "I", "Uh", "Hmm", "Sir" must never be used by the bot.
10. **Context awareness** — if the customer says "Paytm" or "QR", the bot should lock UPI flow and NEVER ask "UPI ya Card?" — check this.

IMPORTANT: Return ONLY the JSON object. No markdown, no explanation, no code fences. Just raw valid JSON.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_calls(filepath: str) -> list[dict]:
    """Load call data from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_transcript(transcription: list[dict]) -> str:
    """Convert transcription array into a readable text format."""
    lines = []
    for i, turn in enumerate(transcription, 1):
        role = "Agent (Bot)" if turn["role"] == "assistant" else "User (Customer)"
        lines.append(f"Turn {i} [{role}]: {turn['content']}")
    return "\n".join(lines)


def download_audio(url: str, output_dir: str) -> str:
    """Download audio file from URL and return local path."""
    filename = url.split("/")[-1]
    filepath = os.path.join(output_dir, filename)

    print(f"    Downloading audio: {filename}")
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"    Downloaded: {size_mb:.2f} MB")
    return filepath


def build_prompt(transcript_text: str) -> str:
    """Build the full analysis prompt with transcript inserted."""
    return ANALYSIS_PROMPT_TEMPLATE.format(
        bot_prompt=BOT_SYSTEM_PROMPT,
        transcript=transcript_text,
    )


def analyze_call(client: genai.Client, audio_path: str, prompt: str) -> dict:
    """Send audio + prompt to Gemini and get the analysis JSON back."""

    # Upload audio file
    print("    Uploading audio to Gemini...")
    uploaded_file = client.files.upload(file=audio_path)
    print(f"    Upload complete: {uploaded_file.name}")

    # Wait until the file reaches ACTIVE state (not just exits PROCESSING).
    # On some environments the file goes PROCESSING -> ACTIVE; we must wait
    # for ACTIVE explicitly, otherwise generate_content gets 400 INVALID_ARGUMENT.
    max_wait = 120  # seconds
    waited = 0
    while uploaded_file.state.name != "ACTIVE":
        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"Audio file processing failed: {uploaded_file.name}")
        if waited >= max_wait:
            raise RuntimeError(
                f"Timed out waiting for file {uploaded_file.name} to become ACTIVE "
                f"(current state: {uploaded_file.state.name})"
            )
        print(f"    Waiting for file to become ACTIVE (current: {uploaded_file.state.name})...")
        time.sleep(5)
        waited += 5
        uploaded_file = client.files.get(name=uploaded_file.name)

    print(f"    File is ACTIVE. Sending to Gemini for analysis...")

    try:
        # Call Gemini — do NOT use response_mime_type="application/json" without
        # a response_schema on gemini-2.5-pro; newer SDK versions reject it with
        # 400 INVALID_ARGUMENT. We instruct the model to return JSON in the prompt
        # instead and parse it ourselves.
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type="audio/wav",
                        ),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=8192,
            ),
        )
    finally:
        # Always clean up the uploaded file to avoid Gemini File API quota buildup
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

    # Parse JSON from response
    raw_text = response.text.strip()

    # Try to parse directly
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        # Strip markdown code fences if the model wrapped the JSON
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        return json.loads(raw_text.strip())


def save_results(results: list[dict], filepath: str):
    """Save analysis results to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {filepath}")


def save_results_csv(results: list[dict], filepath: str):
    """Flatten nested analysis JSON and save as CSV."""

    def safe(obj, *keys, default=""):
        """Safely traverse nested dicts/lists."""
        cur = obj
        for k in keys:
            if isinstance(cur, dict):
                cur = cur.get(k, default)
            elif isinstance(cur, list) and isinstance(k, int) and k < len(cur):
                cur = cur[k]
            else:
                return default
        return cur if cur is not None else default

    def join_list(obj, *keys, sep=" | "):
        """Get a list from nested path and join as string."""
        val = safe(obj, *keys, default=[])
        if isinstance(val, list):
            return sep.join(json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v) for v in val)
        return str(val)

    # CSV column definitions
    CSV_COLUMNS = [
        "call_index",
        "recording_url",
        "transcript_turns",
        "error",
        # call_metadata
        "phone_number",
        "call_duration_approx",
        "transaction_type_detected",
        "bucket_assigned",
        "docket_number",
        # prompt violation
        "prompt_violation",
        "prompt_violations_detail",
        # violation explanation
        "violation_summary",
        # call handled
        "call_handled",
        "call_handled_reasoning",
        # handling details
        "what_went_right",
        "what_went_wrong",
        # STT issues
        "stt_issues_found",
        "stt_issues_detail",
        "stt_summary",
        # STT fixable
        "stt_fixable_issues",
        "stt_unfixable_issues",
        # known issues
        "A_amount_comprehension_found",
        "A_amount_comprehension_details",
        "A_final_amount_correct",
        "B_number_lakhs_found",
        "B_number_lakhs_instances",
        "C_gender_mismatch_found",
        "C_gender_mismatch_instances",
        "D_bot_repetition_found",
        "D_bot_repetition_phrases",
        "E_stt_number_failures_found",
        "E_stt_number_failures_details",
        "F_comprehension_lag_found",
        "F_comprehension_lag_instances",
        # scores
        "score_prompt_adherence",
        "score_call_handling",
        "score_stt_quality",
        "score_bot_comprehension",
        "score_customer_experience",
        "score_overall",
        # actionable fixes
        "prompt_fixes",
        "stt_fixes",
        "tts_fixes",
        "flow_fixes",
        # escalation
        "client_escalation_worthy",
        "client_escalation_reason",
    ]

    rows = []
    for r in results:
        a = r.get("analysis", {})
        meta = safe(a, "call_metadata", default={})
        ki = safe(a, "7_known_issue_check", default={})
        scores = safe(a, "8_overall_scores", default={})
        fixes = safe(a, "9_actionable_fixes", default={})
        esc = safe(a, "10_client_escalation_worthy", default={})

        row = {
            "call_index": r.get("call_index", ""),
            "recording_url": r.get("recording_url", ""),
            "transcript_turns": r.get("transcript_turns", ""),
            "error": r.get("error", ""),
            # metadata
            "phone_number": safe(meta, "phone_number"),
            "call_duration_approx": safe(meta, "call_duration_approx"),
            "transaction_type_detected": safe(meta, "transaction_type_detected"),
            "bucket_assigned": safe(meta, "bucket_assigned"),
            "docket_number": safe(meta, "docket_number"),
            # prompt violations
            "prompt_violation": safe(a, "1_prompt_violation", "answer"),
            "prompt_violations_detail": join_list(a, "1_prompt_violation", "violations"),
            # violation explanation
            "violation_summary": safe(a, "2_violation_explanation", "summary"),
            # call handled
            "call_handled": safe(a, "3_call_handled", "answer"),
            "call_handled_reasoning": safe(a, "3_call_handled", "reasoning"),
            # handling details
            "what_went_right": safe(a, "4_handling_details", "what_went_right"),
            "what_went_wrong": safe(a, "4_handling_details", "what_went_wrong"),
            # STT
            "stt_issues_found": safe(a, "5_stt_issues", "answer"),
            "stt_issues_detail": join_list(a, "5_stt_issues", "issues"),
            "stt_summary": safe(a, "5_stt_issues", "summary"),
            # STT fixable
            "stt_fixable_issues": join_list(a, "6_stt_fixable_via_prompt", "fixable_issues"),
            "stt_unfixable_issues": join_list(a, "6_stt_fixable_via_prompt", "unfixable_issues"),
            # known issues
            "A_amount_comprehension_found": safe(ki, "A_amount_comprehension", "found"),
            "A_amount_comprehension_details": safe(ki, "A_amount_comprehension", "details"),
            "A_final_amount_correct": safe(ki, "A_amount_comprehension", "final_amount_correct"),
            "B_number_lakhs_found": safe(ki, "B_number_reading_in_lakhs", "found"),
            "B_number_lakhs_instances": join_list(ki, "B_number_reading_in_lakhs", "instances"),
            "C_gender_mismatch_found": safe(ki, "C_gender_mismatch", "found"),
            "C_gender_mismatch_instances": join_list(ki, "C_gender_mismatch", "instances"),
            "D_bot_repetition_found": safe(ki, "D_bot_repetition", "found"),
            "D_bot_repetition_phrases": join_list(ki, "D_bot_repetition", "repeated_phrases"),
            "E_stt_number_failures_found": safe(ki, "E_stt_number_failures", "found"),
            "E_stt_number_failures_details": safe(ki, "E_stt_number_failures", "details"),
            "F_comprehension_lag_found": safe(ki, "F_comprehension_lag", "found"),
            "F_comprehension_lag_instances": safe(ki, "F_comprehension_lag", "instances"),
            # scores
            "score_prompt_adherence": safe(scores, "prompt_adherence"),
            "score_call_handling": safe(scores, "call_handling"),
            "score_stt_quality": safe(scores, "stt_quality"),
            "score_bot_comprehension": safe(scores, "bot_comprehension"),
            "score_customer_experience": safe(scores, "customer_experience"),
            "score_overall": safe(scores, "overall"),
            # fixes
            "prompt_fixes": join_list(fixes, "prompt_fixes"),
            "stt_fixes": join_list(fixes, "stt_fixes"),
            "tts_fixes": join_list(fixes, "tts_fixes"),
            "flow_fixes": join_list(fixes, "flow_fixes"),
            # escalation
            "client_escalation_worthy": safe(esc, "answer"),
            "client_escalation_reason": safe(esc, "reason"),
        }
        rows.append(row)

    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved to {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_single_call(client, call, call_index, total_calls, temp_dir, results, results_lock):
    """Process a single call: download, analyze, save. Thread-safe."""
    print(f"{'='*60}")
    print(f"[Worker] Analyzing call {call_index}/{total_calls}")
    print(f"  Recording: {call['recording_url']}")
    print(f"  Transcript turns: {len(call['transcription'])}")

    try:
        # Format transcript
        transcript_text = format_transcript(call["transcription"])

        # Download audio
        audio_path = download_audio(call["recording_url"], temp_dir)

        # Build prompt
        prompt = build_prompt(transcript_text)

        # Analyze with Gemini
        analysis = analyze_call(client, audio_path, prompt)

        # Add metadata about which call this is
        result = {
            "call_index": call_index,
            "recording_url": call["recording_url"],
            "transcript_turns": len(call["transcription"]),
            "analysis": analysis,
        }

        # Thread-safe save
        with results_lock:
            results.append(result)
            save_results(results, OUTPUT_FILE)
            save_results_csv(results, OUTPUT_CSV)

        print(f"  ✓ Call {call_index} analyzed successfully.\n")

        # Clean up audio file
        try:
            os.remove(audio_path)
        except OSError:
            pass

    except Exception as e:
        print(f"  ✗ ERROR analyzing call {call_index}: {e}")
        traceback.print_exc()

        # Save error result so we can track failures
        with results_lock:
            results.append({
                "call_index": call_index,
                "recording_url": call["recording_url"],
                "error": str(e),
            })
            save_results(results, OUTPUT_FILE)
            save_results_csv(results, OUTPUT_CSV)
        print()


def main():
    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Please set the GEMINI_API_KEY environment variable.")
        print("  set GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Load call data
    print(f"Loading calls from {INPUT_FILE}...")
    calls = load_calls(INPUT_FILE)
    print(f"Found {len(calls)} calls to analyze.")
    print(f"Running with {PARALLEL_WORKERS} parallel workers.\n")

    # Create temp directory for audio downloads
    temp_dir = tempfile.mkdtemp(prefix="call_analysis_")
    print(f"Temp directory for audio: {temp_dir}\n")

    results = []

    # If partial results exist, resume from where we left off
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Resuming from call #{len(results) + 1} (found {len(results)} existing results)\n")
        except (json.JSONDecodeError, Exception):
            results = []

    start_index = len(results)
    remaining_calls = calls[start_index:]

    if not remaining_calls:
        print("All calls already analyzed. Nothing to do.")
    else:
        results_lock = threading.Lock()
        total_calls = len(calls)

        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = []
            for offset, call in enumerate(remaining_calls):
                call_index = start_index + offset + 1
                future = executor.submit(
                    process_single_call,
                    client, call, call_index, total_calls,
                    temp_dir, results, results_lock,
                )
                futures.append(future)

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

    # Final summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"  Total calls: {len(calls)}")
    print(f"  Analyzed: {sum(1 for r in results if 'analysis' in r)}")
    print(f"  Errors: {sum(1 for r in results if 'error' in r)}")
    print(f"  Results saved to: {OUTPUT_FILE}")
    print(f"  CSV saved to: {OUTPUT_CSV}")

    # Cleanup temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
