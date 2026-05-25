import os
import logging
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from bson import ObjectId
from backend.database.mongo import users, mongo_db, farmers as farmers_col

log = logging.getLogger(__name__)

sms_bp = Blueprint('sms', __name__, url_prefix='/api/sms')

# Farmers collection (MongoDB)
farmers_col = mongo_db["farmers"]


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def to_oid(id_str):
    try:
        return ObjectId(id_str)
    except Exception:
        return None


def check_admin():
    user_id = get_jwt_identity()
    oid = to_oid(user_id)
    user = users.find_one({"_id": oid if oid else user_id})
    return user if user and user.get("is_admin") else None


def serialize_farmer(f):
    f["_id"] = str(f["_id"])
    if isinstance(f.get("last_sms_at"), datetime):
        f["last_sms_at"] = f["last_sms_at"].isoformat()
    if isinstance(f.get("created_at"), datetime):
        f["created_at"] = f["created_at"].isoformat()
    return f


# ════════════════════════════════════════════════════════════════════════════
#  SMS SENDER
# ════════════════════════════════════════════════════════════════════════════

def send_sms(phone: str, message: str) -> dict:
    provider = os.getenv('SMS_PROVIDER', 'textbelt').lower()
    if not phone.startswith('+'):
        phone = '+977' + phone.lstrip('0')
    if provider == 'twilio':
        return _send_twilio(phone, message)
    elif provider == 'sparrow':
        return _send_sparrow(phone, message)
    else:
        return _send_textbelt(phone, message)


def _send_twilio(phone, message):
    try:
        from twilio.rest import Client
        client = Client(os.getenv('TWILIO_SID'), os.getenv('TWILIO_TOKEN'))
        msg = client.messages.create(
            body=message, from_=os.getenv('TWILIO_FROM'), to=phone)
        return {'success': True, 'info': msg.sid}
    except Exception as e:
        log.error(f'Twilio error to {phone}: {e}')
        return {'success': False, 'info': str(e)}


def _send_sparrow(phone, message):
    try:
        import requests
        local = phone.replace('+977', '')
        r = requests.post('http://api.sparrowsms.com/v2/sms/', data={
            'token': os.getenv('SPARROW_TOKEN'),
            'from':  os.getenv('SPARROW_FROM', 'Cardamom'),
            'to':    local,
            'text':  message,
        }, timeout=10)
        d = r.json()
        return {'success': d.get('response_code') == 200, 'info': str(d)}
    except Exception as e:
        log.error(f'Sparrow SMS error to {phone}: {e}')
        return {'success': False, 'info': str(e)}


def _send_textbelt(phone, message):
    try:
        import requests
        r = requests.post('https://textbelt.com/text', {
            'phone':   phone,
            'message': message,
            'key':     os.getenv('TEXTBELT_KEY', 'textbelt'),
        }, timeout=10)
        d = r.json()
        return {'success': d.get('success', False), 'info': str(d)}
    except Exception as e:
        return {'success': False, 'info': str(e)}


# ════════════════════════════════════════════════════════════════════════════
#  MESSAGE BUILDER
# ════════════════════════════════════════════════════════════════════════════

RISK_NE    = {'High': 'उच्च', 'Medium': 'मध्यम', 'Low': 'कम'}
RISK_EMOJI = {'High': '🔴',   'Medium': '🟡',     'Low': '🟢'}
DISEASE_NE = {
    'chirke':      'चिर्के रोग',
    'chhirke':     'चिर्के रोग',
    'leaf_blight': 'पात झुल्सा',
    'healthy':     'स्वस्थ',
}


def build_weekly_sms(farmer, weather_data, recent_predictions, lang='ne'):
    district_name = (farmer.get('district') or 'ilam').title()
    now = datetime.utcnow().strftime('%Y-%m-%d')

    if lang == 'ne':
        risk       = weather_data.get('risk', {}).get('overall', 'Low')
        risk_ne    = RISK_NE.get(risk, 'कम')
        risk_emoji = RISK_EMOJI.get(risk, '🟢')
        risks      = weather_data.get('risk', {}).get('risks', [])

        top_risk_line = ''
        for r in risks:
            if r['level'] == 'High':
                top_risk_line = f"\n⚠️ {r['disease']}: {r['action'][:50]}"
                break

        pred_line = ''
        if recent_predictions:
            last = recent_predictions[0]
            disease_name = DISEASE_NE.get(last['disease'], last['disease'])
            pred_line = f"\n📊 पछिल्लो स्क्यान: {disease_name} ({last['confidence']}%)"

        msg = (
            f"🌿 अलैँची साप्ताहिक सतर्कता ({now})\n"
            f"📍 {district_name}: {weather_data.get('temperature','?')}°C, "
            f"आर्द्रता {weather_data.get('humidity','?')}%\n"
            f"{risk_emoji} रोग जोखिम: {risk_ne}"
            f"{top_risk_line}{pred_line}\n"
            f"cardamomdx.com मा जानकारीका लागि"
        )
    else:
        risk  = weather_data.get('risk', {}).get('overall', 'Low')
        risks = weather_data.get('risk', {}).get('risks', [])
        top   = next((r for r in risks if r['level'] == 'High'), None)

        pred_line = ''
        if recent_predictions:
            last = recent_predictions[0]
            pred_line = f"\nLast scan: {last['disease'].replace('_',' ')} ({last['confidence']}%)"

        msg = (
            f"🌿 CardamomDx Weekly Alert ({now})\n"
            f"📍 {district_name}: {weather_data.get('temperature','?')}°C\n"
            f"{RISK_EMOJI.get(risk,'🟢')} Disease risk: {risk}"
            + (f"\n⚠️ {top['disease']}: {top['action'][:50]}" if top else '')
            + pred_line + "\ncardamomdx.com"
        )

    return msg[:160]


# ════════════════════════════════════════════════════════════════════════════
#  WEEKLY JOB
# ════════════════════════════════════════════════════════════════════════════

def run_weekly_alerts(app):
    with app.app_context():
        from backend.database.mongo import predictions as predictions_col

        active_farmers = list(farmers_col.find({"is_active": True}))
        if not active_farmers:
            log.info('Weekly SMS: no active farmers')
            return

        log.info(f'Weekly SMS: sending to {len(active_farmers)} farmers')

        weather_cache = {}
        sent = failed = 0

        for farmer in active_farmers:
            try:
                district = farmer.get('district') or 'ilam'
                if district not in weather_cache:
                    weather_cache[district] = _fetch_weather(app, district)
                weather = weather_cache[district]

                week_ago = datetime.utcnow() - timedelta(days=7)
                recent = list(predictions_col.find(
                    {"created_at": {"$gte": week_ago}}
                ).sort("created_at", -1).limit(3))
                for p in recent:
                    p["_id"] = str(p["_id"])

                message = build_weekly_sms(farmer, weather, recent, farmer.get('language', 'ne'))
                result  = send_sms(farmer['phone'], message)

                if result['success']:
                    farmers_col.update_one(
                        {"_id": farmer["_id"]},
                        {"$set": {"last_sms_at": datetime.utcnow()}}
                    )
                    sent += 1
                    log.info(f'SMS sent to {farmer["phone"]} ({farmer["name"]})')
                else:
                    failed += 1
                    log.warning(f'SMS failed to {farmer["phone"]}: {result["info"]}')

            except Exception as e:
                failed += 1
                log.error(f'Error sending to farmer {farmer.get("_id")}: {e}')

        log.info(f'Weekly SMS complete: {sent} sent, {failed} failed')
        return {'sent': sent, 'failed': failed}


def _fetch_weather(app, district):
    try:
        with app.test_client() as client:
            r = client.get(f'/api/weather?district={district}')
            return r.get_json() or {}
    except Exception as e:
        log.error(f'Weather fetch failed for {district}: {e}')
        return {'temperature': '?', 'humidity': '?', 'rain_mm': 0,
                'risk': {'overall': 'Medium', 'risks': []}, 'demo': True}


# ════════════════════════════════════════════════════════════════════════════
#  SCHEDULER
# ════════════════════════════════════════════════════════════════════════════

def start_scheduler(app):
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=run_weekly_alerts,
            trigger=CronTrigger(day_of_week='sun', hour=2, minute=15, timezone='UTC'),
            args=[app],
            id='weekly_sms_alert',
            replace_existing=True,
            name='Weekly SMS Disease Alert',
        )
        scheduler.start()
        log.info('✅ SMS scheduler started — weekly alerts every Sunday 8:00 AM NST')
        return scheduler
    except ImportError:
        log.warning('APScheduler not installed. Run: pip install apscheduler')
        return None
    except Exception as e:
        log.error(f'Scheduler failed to start: {e}')
        return None


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC ROUTES
# ════════════════════════════════════════════════════════════════════════════

@sms_bp.route('/register', methods=['POST'])
def register_farmer():
    data     = request.get_json(silent=True) or {}
    name     = (data.get('name') or '').strip()
    phone    = (data.get('phone') or '').strip()
    district = (data.get('district') or 'ilam').strip().lower()
    language = (data.get('language') or 'ne').strip().lower()

    if not name or not phone:
        return jsonify({'error': 'Name and phone number are required'}), 400

    clean = phone.replace(' ', '').replace('-', '').lstrip('+977').lstrip('0')
    if not (clean.isdigit() and 9 <= len(clean) <= 10):
        return jsonify({'error': 'Enter a valid Nepal phone number (e.g. 9812345678)'}), 400

    existing = farmers_col.find_one({"phone": phone})
    if existing:
        if existing.get('is_active'):
            return jsonify({'message': f'✅ {existing["name"]}, you are already registered!'}), 200
        farmers_col.update_one({"phone": phone}, {"$set": {
            "is_active": True, "name": name, "district": district, "language": language
        }})
        return jsonify({'message': '✅ Welcome back! SMS alerts re-activated.'}), 200

    result = farmers_col.insert_one({
        "name": name, "phone": phone, "district": district,
        "language": language, "is_active": True,
        "last_sms_at": None, "created_at": datetime.utcnow(),
    })

    welcome_ne = f"🌿 नमस्ते {name}! अलैँची साप्ताहिक सतर्कता सेवामा स्वागत छ। हरेक आइतबार रोग र मौसम अलर्ट पाउनुहुनेछ। - CardamomDx"
    welcome_en = f"🌿 Hi {name}! You're registered for CardamomDx weekly disease & weather alerts every Sunday. - cardamomdx.com"
    send_sms(phone, (welcome_ne if language == 'ne' else welcome_en)[:160])

    return jsonify({
        'message': '✅ Registered successfully! You will receive weekly SMS alerts every Sunday morning.',
        'farmer_id': str(result.inserted_id),
    }), 201


@sms_bp.route('/unregister', methods=['POST'])
def unregister_farmer():
    data  = request.get_json(silent=True) or {}
    phone = (data.get('phone') or '').strip()
    if not phone:
        return jsonify({'error': 'Phone number is required'}), 400

    farmer = farmers_col.find_one({"phone": phone})
    if not farmer:
        return jsonify({'error': 'Phone number not found in our system'}), 404

    farmers_col.update_one({"phone": phone}, {"$set": {"is_active": False}})
    return jsonify({'message': '✅ You have been unregistered from SMS alerts.'}), 200


# ════════════════════════════════════════════════════════════════════════════
#  ADMIN ROUTES
# ════════════════════════════════════════════════════════════════════════════

@sms_bp.route('/farmers', methods=['GET'])
@jwt_required()
def list_farmers():
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403

    page     = request.args.get('page',  1,  type=int)
    per_page = request.args.get('limit', 10, type=int)
    district = request.args.get('district', None)

    query = {}
    if district:
        query["district"] = district

    skip  = (page - 1) * per_page
    total = farmers_col.count_documents(query)
    total_active = farmers_col.count_documents({"is_active": True})

    items = [serialize_farmer(f) for f in
             farmers_col.find(query).sort("created_at", -1).skip(skip).limit(per_page)]

    return jsonify({
        'farmers':      items,
        'total':        total,
        'total_active': total_active,
        'page':         page,
        'pages':        (total + per_page - 1) // per_page,
    }), 200


@sms_bp.route('/farmers/<fid>', methods=['DELETE'])
@jwt_required()
def delete_farmer(fid):
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403

    oid    = to_oid(fid)
    farmer = farmers_col.find_one({"_id": oid if oid else fid})
    if not farmer:
        return jsonify({'error': 'Farmer not found'}), 404

    farmers_col.delete_one({"_id": farmer["_id"]})
    return jsonify({'message': f'Farmer {farmer["name"]} removed'}), 200


@sms_bp.route('/send-now', methods=['POST'])
@jwt_required()
def send_now():
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403

    from backend.database.mongo import predictions as predictions_col

    data     = request.get_json(silent=True) or {}
    district = data.get('district', None)

    query = {"is_active": True}
    if district:
        query["district"] = district

    active_farmers = list(farmers_col.find(query))
    if not active_farmers:
        return jsonify({'error': 'No active farmers to send to'}), 400

    weather_cache = {}
    sent = failed = 0

    for farmer in active_farmers:
        try:
            d = farmer.get('district') or 'ilam'
            if d not in weather_cache:
                weather_cache[d] = _fetch_weather(current_app._get_current_object(), d)
            weather = weather_cache[d]

            recent = list(predictions_col.find().sort("created_at", -1).limit(3))
            for p in recent:
                p["_id"] = str(p["_id"])

            message = build_weekly_sms(farmer, weather, recent, farmer.get('language', 'ne'))
            result  = send_sms(farmer['phone'], message)

            if result['success']:
                farmers_col.update_one(
                    {"_id": farmer["_id"]},
                    {"$set": {"last_sms_at": datetime.utcnow()}}
                )
                sent += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            log.error(f'send-now error for farmer {farmer.get("_id")}: {e}')

    return jsonify({'message': f'Sent to {sent} farmer(s).', 'sent': sent, 'failed': failed}), 200


@sms_bp.route('/send-custom', methods=['POST'])
@jwt_required()
def send_custom():
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403

    data    = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'error': 'Message is required'}), 400
    if len(message) > 160:
        return jsonify({'error': f'Message too long ({len(message)}/160 characters)'}), 400

    active_farmers = list(farmers_col.find({"is_active": True}))
    if not active_farmers:
        return jsonify({'error': 'No active farmers'}), 400

    sent = failed = 0
    for farmer in active_farmers:
        result = send_sms(farmer['phone'], message)
        if result['success']:
            farmers_col.update_one(
                {"_id": farmer["_id"]},
                {"$set": {"last_sms_at": datetime.utcnow()}}
            )
            sent += 1
        else:
            failed += 1

    return jsonify({'message': f'Sent to {sent} farmer(s).', 'sent': sent, 'failed': failed}), 200


@sms_bp.route('/stats', methods=['GET'])
@jwt_required()
def sms_stats():
    if not check_admin():
        return jsonify({'error': 'Admin access required'}), 403

    total  = farmers_col.count_documents({})
    active = farmers_col.count_documents({"is_active": True})

    by_district = {}
    for f in farmers_col.find({"is_active": True}, {"district": 1}):
        d = f.get("district") or "unknown"
        by_district[d] = by_district.get(d, 0) + 1

    last = farmers_col.find_one(
        {"last_sms_at": {"$ne": None}},
        sort=[("last_sms_at", -1)]
    )

    return jsonify({
        'total_farmers':  total,
        'active_farmers': active,
        'by_district':    by_district,
        'last_sent_at':   last["last_sms_at"].isoformat() if last and last.get("last_sms_at") else None,
        'provider':       os.getenv('SMS_PROVIDER', 'textbelt'),
    }), 200