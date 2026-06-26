import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_INPUT_JSON = Path("data/transcripts/transcripts.json")
DEFAULT_OUTPUT_JSON = Path("data/preprocessed/preprocessed.json")
DEFAULT_REVIEW_CSV = Path("data/preprocessed/review_preprocessed_transcripts.csv")

LOW_CONFIDENCE_THRESHOLD = 0.75
VERY_SHORT_CALL_WORDS = 5
HIGH_CLEANING_RATIO_THRESHOLD = 0.55

LOW_WORD_CONFIDENCE_THRESHOLD = 0.40
VERY_LOW_WORD_CONFIDENCE_THRESHOLD = 0.20
HIGH_LOW_CONFIDENCE_WORD_RATIO = 0.30
MAX_EXPORTED_CONFIDENCE_WORDS = 30
MIN_LOW_CONFIDENCE_NOISE_WORD_LEN = 4
REMOVE_LOW_CONFIDENCE_NOISE_WORDS = False

BOILERPLATE_PATTERNS: list[tuple[str, str]] = [
    (r"\bуведомляем\s+о\s+записи\s+разговора\b", "recording_notice_ru"),
    (r"\b(?:этот|данный|ваш)?\s*разговор\s+(?:будет\s+)?записыва(?:е|и)тся\b", "recording_notice_ru"),
    (r"\bзвонок\s+(?:будет\s+)?записыва(?:е|и)тся\b", "recording_notice_ru"),
    (r"\bәңгіменің\s+жазылуы\s+туралы\s+ескертеміз\b", "recording_notice_kk"),
    (r"\bсөйлесудің\s+жазылуы\s+туралы\s+ескертеміз\b", "recording_notice_kk"),
    (r"\bқоңырау\s+жазылады\b", "recording_notice_kk"),
]

OPERATOR_PATTERNS: list[tuple[str, str]] = [
    (r"\bздравствуйте\b", "greeting_ru"),
    (r"\bдобрый\s+(?:день|вечер|утро)\b", "greeting_ru"),
    (r"\bдевушка\s+здравствуйте\b", "greeting_ru"),
    (r"\bподскажите\s+пожалуйста\b", "polite_phrase_ru"),
    (r"\bдевушка\b", "address_word_ru"),
    (r"\bпожалуйста\b", "polite_phrase_ru"),
    (r"\bвот\b", "filler_word_ru"),
    (r"\bчем\s+могу\s+помочь\b", "operator_intro_ru"),
    (r"\bменя\s+зовут\s+[а-яёәғқңөұүһі]+\b", "operator_intro_ru"),
    (r"\bя\s+слушаю\b", "operator_intro_ru"),
    (r"\bслушаю\s+вас\b", "operator_intro_ru"),

    (r"\bсаламатсыз\s+ба\b", "greeting_kk"),
    (r"\bсәлеметсіз\s+бе\b", "greeting_kk"),
    (r"\bтыңдап\s+тұрмын\b", "operator_intro_kk"),
    (r"\bтыңдап\s+отырмын\b", "operator_intro_kk"),
    (r"\bқосымша\s+сұрақтарыңыз\s+бар\s+ма\b", "operator_closing_kk"),
]

CLOSING_PATTERNS: list[tuple[str, str]] = [
    (r"\bспасибо\s+большое\b", "closing_ru"),
    (r"\bспасибо\b", "closing_ru"),
    (r"\bвсего\s+доброго\b", "closing_ru"),
    (r"\bвсего\s+добра\b", "closing_ru"),
    (r"\bдо\s+свидания\b", "closing_ru"),
    (r"\bрахмет\b", "closing_kk"),
    (r"\bрақмет\b", "closing_kk"),
    (r"\bсау\s+болыңыз\b", "closing_kk"),
    (r"\bаман\s+болыңыз\b", "closing_kk"),
    (r"\bқоңырау\s+шалғаныңызға\s+рақмет\b", "closing_kk"),
]

VERIFICATION_PATTERNS: list[tuple[str, str]] = [
    (r"\bдавайте\s+проверим\b.*?(?:фамилия|имя|отчество|дата\s+рождения|кодовое\s+слово|иин).*$", "verification_tail_ru"),
    (r"\b(?:ваша|ваш)?\s*полная\s+фамилия\s+имя\s+отчество\b.*$", "full_name_request_ru"),
    (r"\bфамилия\s+имя\s+отчество\b.*$", "full_name_request_ru"),
    (r"\b(?:ваша|ваш)?\s*дата\s+рождения\b.*$", "birth_date_ru"),
    (r"\bкодовое\s+слово\b.*$", "code_word_ru"),
    (r"\bв\s+таком\s+случае\s+ваш\b.*$", "verification_tail_ru"),

    (r"\bтуған\s+(?:жылыңыз|жылым|жуыңыз|күніңіз)\b.*$", "birth_date_kk"),
    (r"\bқұпия\s+сөзіңіз\b.*$", "code_word_kk"),
    (r"\bкод\s+сөзіңіз\b.*$", "code_word_kk"),
]

ASR_GARBAGE_PATTERNS: list[tuple[str, str]] = [
    (r"\bдоктор\s+быка\b", "asr_bank_name_noise"),
    (r"\bсэр\s+быка\b", "asr_bank_name_noise"),
    (r"\bщербак\s+банка\b", "asr_bank_name_noise"),
    (r"\bпрозрачный\s+старбак\s+в\s+ангаре\b", "asr_bank_name_noise"),
    (r"\bдэвидтен\b", "asr_noise"),
    (r"\bбербок\s+шоу\b", "asr_noise"),
    (r"\bгейминатор\b", "asr_noise"),
    (r"\bвидео\s+пробирка", "asr_video_check_noise"),
]

PROBLEM_KEYWORDS_RU = [
    # по аудио
    "не могу", "не получается", "не работает", "не проходит", "не приходит",
    "ошибка", "забыл", "забыли", "войти", "логин", "пароль", "аккаунт",
    "код", "смс", "номер", "телефон", "карта", "карточка", "карту",
    "перевод", "переводы", "отправить", "деньги", "золотой короне",
    "кредит", "кредитов", "платеж", "платёж", "оплата", "оплате",
    "счёт", "счет", "пополнить", "пополнил", "спишутся", "списаться",
    "списали", "удержания", "просрочка", "отсрочка", "лимит", "превышен",
    "отделение", "работает", "рабочее время", "выходной", "суббота", "воскресенье",
    "напоминание", "система", "банк",

    # доступ и авторизация
    "не могу войти", "не могу зайти", "зайти не могу",
    "неверный логин", "неверный пароль", "восстановить пароль",
    "сбросить пароль", "заблокирован аккаунт", "заблокировали доступ",
    "сессия истекла", "требуется повторный вход", "двухфакторная аутентификация",
    "вход через госуслуги", "вход по биометрии", "отпечаток пальца", "face id",

    # приложение и интернет-банк
    "приложение не открывается", "приложение вылетает", "приложение зависло",
    "бесконечная загрузка", "чёрный экран", "белый экран", "краш",
    "баг", "глюк", "не загружается", "не обновляется", "ошибка соединения",
    "нет сети", "проверьте подключение", "ошибка сервера", "502", "ошибка 500",
    "технические работы", "регламентные работы", "недоступен сервис",
    "обновите приложение", "устаревшая версия", "версия не поддерживается",
    "ios", "android", "несовместимость", "очистить кэш", "переустановить",
    "мобильный банк не работает", "интернет-банк не грузится", "личный кабинет недоступен",

    # смс и пуш
    "не приходит смс", "не приходит код", "смс задерживается",
    "пуш не приходит", "уведомление не отображается", "нет уведомлений",
    "отключены уведомления", "подключить смс-информирование", "отключить смс",
    "неправильный код подтверждения", "код просрочен",

    # операции и платежи
    "не могу отправить перевод", "перевод не уходит", "перевод завис",
    "не зачислились деньги", "не дошли деньги", "деньги не пришли",
    "ошибочный перевод", "ошибка в реквизитах", "некорректные данные получателя",
    "платёж не прошёл", "отклонена операция", "отказ в проведении",
    "подозрительная операция", "подтверждение операции не проходит",
    "3d secure не работает", "зависла оплата", "двойное списание",
    "списали дважды", "дублирующая транзакция", "неверная сумма",
    "возврат денег", "чарджбэк", "оспорить транзакцию",
    "розыск перевода", "отозвать платёж", "не могу отменить платёж",
    "автоплатёж не сработал", "шаблон платежа удалился",

    # карты (блокировка, перевыпуск, лимиты)
    "карта заблокирована", "заблокировали карту", "временная блокировка",
    "разблокировать карту", "утеряна карта", "украли карту",
    "скомпрометирована", "мошеннические действия", "подозрительная активность",
    "перевыпустить карту", "заказать новую карту", "виртуальная карта",
    "пластиковая карта", "именная карта", "неименная", "срок действия истёк",
    "не принимают карту", "терминал не читает", "чип не работает",
    "магнитная полоса", "бесконтактная оплата не проходит",
    "apple pay не добавляется", "google pay не работает", "mir pay",
    "не могу привязать карту", "токен не создаётся",
    "превышен лимит по карте", "суточный лимит", "месячный лимит",
    "ограничение на снятие", "лимит на переводы", "снять наличные не могу",
    "банкомат не выдал деньги", "банкомат зажевал карту",
    "остаток средств не обновляется", "баланс не отображается",

    # счета и обслуживание
    "открыть счёт", "закрыть счёт", "закрыть карту", "закрыть карточку",
    "годовое обслуживание списали", "комиссия за обслуживание",
    "комиссия за смс", "пакет услуг", "тариф",
    "не пользуюсь картой", "вернуть комиссию", "скрытая комиссия",
    "навязали страховку", "подключили услугу без согласия",
    "автоподписка", "отключить услугу", "отказаться от страховки",

    # кредиты и задолженности
    "не могу погасить кредит", "досрочное погашение", "частичное досрочное",
    "график платежей", "сумма задолженности не совпадает",
    "проценты пересчитали неверно", "ошибочные начисления",
    "просроченная задолженность", "отсрочка платежа", "кредитные каникулы",
    "реструктуризация долга", "рефинансирование", "заявка на кредит",
    "отказ в кредите", "одобрение заявки", "кредитная история испорчена",
    "ошибка в кредитной истории", "коллекторы звонят", "судебные приставы",
    "арест счета", "исполнительный лист", "удержания из зарплаты",

    # переводы (системы, валюта, за границу)
    "золотая корона", "contact", "western union", "юнистрим",
    "корона выплаты", "не могу получить перевод", "код перевода",
    "отправить за границу", "валютный перевод", "swift перевод",
    "санкции не дают отправить", "банк-корреспондент отклонил",
    "заморожен перевод", "комиссия за перевод слишком высокая",
    "курс конвертации невыгодный", "конвертация валюты", "обменять валюту",
    "наличные доллары", "наличные евро", "заказать валюту", "касса без валюты",
    "покупка валюты по бирже", "спред",

    # безопасность и мошенничество
    "звонок из банка", "звонили мошенники", "представились службой безопасности",
    "просили код из смс", "сообщил данные карты", "украли деньги",
    "несанкционированное списание", "не совершал эту операцию",
    "заблокируйте карту срочно", "оспорить мошенническую операцию",
    "данные карты скомпрометированы", "пин-код забыл", "кодовое слово не помню",
    "сменить кодовое слово", "изменить лимит на переводы",

    # общие проблемы и недовольства
    "долго жду ответа", "робот не понимает", "переключите на оператора",
    "оператор сбросил звонок", "плохая связь", "не слышно",
    "предыдущее обращение не решено", "номер обращения", "жалоба",
    "претензия", "заявление", "официальный ответ",
    "технический сбой", "системная ошибка", "обновление системы",
    "перерыв в работе", "деньги зависли", "транзакция в обработке",
    "слишком медленное зачисление", "прошу разобраться",
]


PROBLEM_KEYWORDS_KK = [
    # из аудио
    "несие", "төлем", "төледім", "төлеп", "төлесем", "төлей",
    "шот", "ақша", "ұсталмайды", "ұсталынбайтын",
    "шешілмейді", "демалыс", "мейрам", "күндері", "ертең", "бүгін",
    "қалай", "қай жерден", "хабарласып", "берешек",
    "сұрақ", "тоқтат", "қалпына келтіру", "көшіру",
    "тіркелген",

    # доступ / авторизация
    "кіре алмай жатырмын", "кіру мүмкін емес", "жүйеге кіре алмадым",
    "аккаунтқа кіре алмай тұрмын",
    "логин ұмыттым", "пароль ұмыттым",
    "парольді өзгерту", "аккаунт бұғатталды",
    "сессия аяқталды", "екі факторлы қорғау",
    "биометриямен кіру",
    "Face ID жұмыс істемейді", "отпечаток пальца істемейді",

    # приложение / интернет-банк
    "қосымша ашылмайды", "қосымша жабылып қалады",
    "ескі нұсқа", "версия ескірген",
    "кэшті тазалау", "қайта орнату",
    "интернет-банк ашылмай тұр", "интернет-банк жұмыс істемейді",
    "жеке кабинетке кіре алмаймын",
    "мобильді банк жұмыс істемейді",
    "техникалық жұмыстар", "жүйе қолжетімсіз",

    # смс / код / уведомления
    "смс келмей жатыр", "смс келмеді",
    "код келмеді", "код келмей тұр",
    "пуш келмейді", "пуш-хабарландыру келмейді",
    "хабарландыру келмейді", "хабарландыру жоқ",
    "смс-информирование қосу", "смс-информирование өшіру",

    # переводы / төлемдер / аударымдар === өтпей қалды",
    "аударым өтпей жатыр",
    "ақша жібере алмай тұрмын", "ақша түскен жоқ",
    "төлем өтпеді", "төлем бас тартылды",
    "екі рет ұсталды", "екі рет списали",
    "ақша қайтару", "аударымды тоқтату",
    "аударым коды", "аударым комиссиясы",
    "валюталық аударым", "SWIFT аударым",
    "валюта бағамы", "конвертация жасау",

    # карта
    "карта бұғатталды", "картаны бұғаттаңыз",
    "картам жоғалды", "картамды ұрлаттым",
    "карта скомпрометирована", "мошенники списали",
    "картаны ауыстыру", "виртуалды карта",
    "карта мерзімі өтті",
    "карта оқылмай жатыр", "чип оқылмайды",
    "контактсыз төлем өтпейді", "бесконтактная оплата өтпейді",
    "Apple Pay қосылмайды", "Google Pay істемейді",
    "картаны әмиянға қоса алмаймын", "привязать карта алмаймын",
    "лимиттен асып кеттім", "лимит асып кетті",
    "банкомат ақша бермеді", "банкомат картаны жұтып қойды",
    "баланс көрсетілмейді", "баланс көрінбейді",

    # кредит / несие / қарыз
    "несие төлей алмаймын", "несие өтеу мүмкін емес",
    "мерзімі өткен берешек", "просрочка бойынша",
    "төлем кестесі", "проценты неверно",
    "несие өтінімі", "заявка на кредит бердім",
    "несиеден бас тартты", "отказ в кредите алдым",
    "кредиттік демалыс", "кредитные каникулы сұрау",
    "реструктуризация жасау", "рефинансирование жасау",
    "коллекторлар хабарласып жатыр", "коллекторлар мазалап жатыр",
    "сот орындаушылар келді", "шот бұғатталды пристав",

    # отделение / жұмыс уақыты
    "бөлімше қайда", "ең жақын бөлімше",
    "жұмыс уақыты қандай", "жұмыс кестесі",
    "сенбі күні ашық па", "сенбі жұмыс істей ме",
    "жексенбі ашық па", "жексенбі жұмыс күні ме",
    "мейрам күндері жұмыс істей ме", "демалыс күндері",
    "түскі үзіліс", "түскі ас уақыты",

    # безопасность / мошенничество
    "банктен қоңырау шалды", "банктен хабарласты",
    "алаяқтар хабарласты", "алаяқтар қоңырау шалды",
    "код сұрады", "кодты айттым",
    "карта деректерін сұрады", "данные карты сұрады",
    "ақша ұрланды", "ақша жоғалды",
    "мен жасамаған операция", "менің операциям емес",
    "күмәнді транзакция", "подозрительная операция болды",
    "кодовое слово ұмыттым", "кодовое слово ауыстыру",
    "пин-код ұмыттым", "пин-код білмеймін",

    # общие проблемы / жалпы шағымдар
    "операторға қосыңыз", "операторға ауыстырыңыз",
    "робот түсінбейді", "робот көмектеспейді",
    "байланыс үзілді", "байланыс нашар",
    "өтініш нөмірі", "номер обращения бар",
    "шағым қалдыру", "претензия жазу",
    "техникалық ақау", "жүйелік қате",
    "ақша кідіріп жатыр", "ақша завис",
    "транзакция өңделуде", "операция өңделіп жатыр",
    "өте баяу", "көп күтемін",

    # дополнительные смешанные фразы
    "перевод өтпейді", "платеж өтпеді",
    "счет бұғатталды", "счет арест",
    "приложение ашылмай тұр", "приложение істемейді",
    "оплата өтпеді", "оплата жасай алмадым",
    "деньги түспеді", "деньги қайтару",
    "логин есімде жоқ", "пароль дұрыс емес",
    "смс код керек", "код жарамсыз",
]

PROBLEM_KEYWORDS = PROBLEM_KEYWORDS_RU + PROBLEM_KEYWORDS_KK

# важные слова для проверки confidence

CRITICAL_WORDS = {
    # русский
    "кредит",
    "кредита",
    "кредитов",
    "карта",
    "карту",
    "карточка",
    "лимит",
    "пароль",
    "логин",
    "код",
    "смс",
    "аккаунт",
    "перевод",
    "деньги",
    "счет",
    "счёт",
    "шот",
    "оплата",
    "платеж",
    "платёж",
    "просрочка",
    "отсрочка",
    "удержания",
    "списание",
    "списали",
    "пополнить",
    "пополнил",
    "задолженность",
    "долг",
    "комиссия",
    "банкомат",
    "отделение",
    "мошенники",
    "мошенничество",
    "заблокирована",
    "заблокировали",
    "разблокировать",

    # казахский
    "несие",
    "төлем",
    "төлей",
    "төледім",
    "төлеп",
    "ақша",
    "шот",
    "берешек",
    "қарыз",
    "аударым",
    "карта",
    "лимит",
    "код",
    "смс",
    "логин",
    "пароль",
    "құпия",
    "қосымша",
    "бұғатталды",
    "бұғаттау",
    "алаяқтар",
}

IMPORTANT_WORD_STOPWORDS = {
    # общие русские слова
    "банк",
    "банка",
    "система",
    "номер",
    "телефон",
    "работает",
    "можно",
    "нужно",
    "надо",
    "есть",
    "будет",
    "сейчас",
    "сегодня",
    "завтра",
    "там",
    "тут",
    "это",
    "что",
    "как",
    "где",
    "когда",
    "уже",
    "просто",
    "тоже",

    # общие казахские слова
    "мен",
    "сіз",
    "біз",
    "ол",
    "бұл",
    "сол",
    "осы",
    "бойынша",
    "қалады",
    "болды",
    "болады",
    "керек",
    "дейін",
    "кейін",
    "қалай",
    "қайда",
    "қашан",
    "нөмірі",
    "нөмір",
    "жерден",
    "күндері",
    "бүгін",
    "ертең",
}

EMPTY_PHRASES = {
    "",
    "ну",
    "да",
    "нет",
    "ага",
    "угу",
    "мхм",
    "мм",
    "эм",
    "ээ",
    "эээ",
    "а",
    "о",
    "хорошо",
    "понятно",
    "ясно",
    "ладно",
    "ок",
    "окей",
    "спасибо",
    "спасибо большое",
    "благодарю",
    "до свидания",
    "досвидания",
    "всего доброго",
    "всего хорошего",
    "алло",
    "алло алло",
    "алё",
    "але",
    "меня слышно",
    "вы меня слышите",
    "слышно",
    "не слышно",
    "вас не слышно",
    "я вас не слышу",
    "меня не слышно",
    "плохо слышно",
    "повторите",
    "повторите пожалуйста",
    "что",
    "что что",
    "не понял",
    "не поняла",
    "я ошибся",
    "я ошиблась",
    "ошибся",
    "ошиблась",
    "ошибся номером",
    "ошиблась номером",
    "не туда попал",
    "не туда попала",
    "не туда позвонил",
    "не туда позвонила",
    "случайно набрал",
    "случайно набрала",
    "случайно позвонил",
    "случайно позвонила",
    "перезвоните позже",
    "позвоните позже",
    "я перезвоню",
    "перезвоню позже",
    "сейчас не могу говорить",
    "не могу говорить",
    "уже не актуально",
    "вопрос решен",
    "вопрос уже решен",
    "уже решил",
    "уже решила",
    "уже решили",
    "ничего не нужно",
    "ничего не надо",
    "не надо",
    "не нужно",
    "тишина",
    "гудки",
    "музыка",
    "шум",
    "неразборчиво",
    "автоответчик",
    "голосовая почта",
    "иә",
    "ия",
    "жоқ",
    "жаксы",
    "жақсы",
    "мақұл",
    "түсінікті",
    "рахмет",
    "рақмет",
    "көп рахмет",
    "сау болыңыз",
    "сау болыныз",
    "сау бол",
    "аман болыңыз",
    "аман болыныз",
    "жарайды",
    "болды",
    "алло",
    "естіп тұрсыз ба",
    "естіп тұрмын",
    "естімей тұрмын",
    "тыңдап тұрмын",
    "қайта айтыңыз",
    "қайталаңыз",
    "қайталай аласыз ба",
    "не дедіңіз",
    "не дейсіз",
    "слышно",
    "қате нөмір",
    "қате тердім",
    "қате хабарластым",
    "басқа нөмір",
    "кейін хабарласыңыз",
    "кейін қоңырау шалыңыз",
    "өзім хабарласамын",
    "кейін хабарласамын",
    "қазір сөйлесе алмаймын",
    "сұрақ шешілді",
    "енді керек емес",
    "қажет емес",
    "ештеңе керек емес",
    "жете аламыз",
    "сауыты",
}

NUMBER_WORDS = {
    "ноль",
    "нуль",
    "один",
    "одна",
    "одно",
    "два",
    "две",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
    "тысяча",
    "тысячи",
    "тысяч",
    "миллион",
    "миллиона",
    "миллионов",
    "миллиард",
    "миллиарда",
    "миллиардов",
    "первое",
    "второе",
    "третье",
    "четвертое",
    "четвёртое",
    "пятое",
    "шестое",
    "седьмое",
    "восьмое",
    "девятое",
    "десятое",
    "одиннадцатое",
    "двенадцатое",
    "тринадцатое",
    "четырнадцатое",
    "пятнадцатое",
    "шестнадцатое",
    "семнадцатое",
    "восемнадцатое",
    "девятнадцатое",
    "двадцатое",
    "тридцатое",
    "число",
    "числа",
    "день",
    "дня",
    "дней",
    "неделя",
    "недели",
    "недель",
    "месяц",
    "месяца",
    "месяцев",
    "год",
    "года",
    "лет",
    "январь",
    "января",
    "февраль",
    "февраля",
    "март",
    "марта",
    "апрель",
    "апреля",
    "май",
    "мая",
    "июнь",
    "июня",
    "июль",
    "июля",
    "август",
    "августа",
    "сентябрь",
    "сентября",
    "октябрь",
    "октября",
    "ноябрь",
    "ноября",
    "декабрь",
    "декабря",
    "тенге",
    "тиын",
    "рубль",
    "рубля",
    "рублей",
    "копейка",
    "копейки",
    "копеек",
    "доллар",
    "долларов",
    "процент",
    "процента",
    "процентов",
    "нөл",
    "бір",
    "екі",
    "үш",
    "төрт",
    "бес",
    "алты",
    "жеті",
    "сегіз",
    "тоғыз",
    "он",
    "жиырма",
    "отыз",
    "қырық",
    "елу",
    "алпыс",
    "жетпіс",
    "сексен",
    "тоқсан",
    "жүз",
    "мың",
    "миллион",
    "миллиард",
    "бірінші",
    "екінші",
    "үшінші",
    "төртінші",
    "бесінші",
    "алтыншы",
    "жетінші",
    "сегізінші",
    "тоғызыншы",
    "оныншы",
    "жиырмасыншы",
    "отызыншы",
    "күн",
    "күні",
    "күнге",
    "күндері",
    "күндерінде",
    "апта",
    "аптада",
    "аптаға",
    "ай",
    "айда",
    "айға",
    "жыл",
    "жылы",
    "жылға",
    "қаңтар",
    "ақпан",
    "наурыз",
    "сәуір",
    "мамыр",
    "маусым",
    "шілде",
    "тамыз",
    "қыркүйек",
    "қазан",
    "қараша",
    "желтоқсан",
    "теңге",
    "тиын",
    "пайыз",
}

# функции

def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(path)


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower().replace("ё", "е")

    text = re.sub(
        r"[^a-zа-яәғқңөұүһі0-9\s\[\]_@.+\-]",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def apply_patterns(
    text: str,
    patterns: list[tuple[str, str]],
) -> tuple[str, list[str]]:
    removed = []

    for pattern, reason in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
            removed.append(reason)

    text = re.sub(r"\s+", " ", text).strip()
    return text, removed


def has_problem_signal(text: str) -> bool:
    text = normalize_text(text)
    return any(normalize_text(keyword) in text for keyword in PROBLEM_KEYWORDS)


def count_words(text: str) -> int:
    return len(re.findall(r"[a-zа-яәғқңөұүһі0-9\[\]_]+", text.lower()))


def calculate_cleaning_ratio(raw_word_count: Any, cleaned_word_count: Any) -> float | None:
    if not isinstance(raw_word_count, int):
        return None

    if not isinstance(cleaned_word_count, int):
        return None

    if raw_word_count <= 0:
        return None

    return round(cleaned_word_count / raw_word_count, 4)


def build_important_word_set() -> set[str]:
    important_words = set()

    for word in CRITICAL_WORDS:
        word = normalize_text(word)

        if word and word not in IMPORTANT_WORD_STOPWORDS:
            important_words.add(word)

    for keyword in PROBLEM_KEYWORDS:
        keyword = normalize_text(keyword)
        words = keyword.split()

        # короткие однословные ключи можно считать важными
        if len(words) == 1:
            word = words[0]

            if (
                len(word) >= 4
                and word not in IMPORTANT_WORD_STOPWORDS
                and word not in EMPTY_PHRASES
                and word not in NUMBER_WORDS
            ):
                important_words.add(word)

    return important_words


IMPORTANT_WORDS = build_important_word_set()


def is_important_word(word: str) -> bool:
    word = normalize_text(word)

    if not word:
        return False

    return word in IMPORTANT_WORDS

def format_confidence_words(words: list[dict[str, Any]]) -> str:
    formatted = []

    for item in words:
        word = item.get("word")
        conf = item.get("conf")

        if isinstance(conf, (int, float)):
            formatted.append(f"{word}:{conf:.4f}")
        else:
            formatted.append(str(word))

    return "; ".join(formatted)


def analyze_word_confidences(item: dict[str, Any]) -> dict[str, Any]:
    words = item.get("words") or []

    total = 0
    low_count = 0
    very_low_count = 0

    low_confidence_important_words = []
    very_low_confidence_words = []
    removable_low_confidence_noise_words = []

    for word_item in words:
        word = normalize_text(str(word_item.get("word", "")))
        conf = word_item.get("conf")

        if not word or not isinstance(conf, (int, float)):
            continue

        total += 1

        word_info = {
            "word": word,
            "conf": round(float(conf), 4),
        }

        if conf < LOW_WORD_CONFIDENCE_THRESHOLD:
            low_count += 1

            if is_important_word(word):
                low_confidence_important_words.append(word_info)

        if conf < VERY_LOW_WORD_CONFIDENCE_THRESHOLD:
            very_low_count += 1
            very_low_confidence_words.append(word_info)

            if (
                len(word) >= MIN_LOW_CONFIDENCE_NOISE_WORD_LEN
                and not word.isdigit()
                and word not in NUMBER_WORDS
                and word not in EMPTY_PHRASES
                and not is_important_word(word)
            ):
                removable_low_confidence_noise_words.append(word_info)

    low_ratio = None
    if total > 0:
        low_ratio = round(low_count / total, 4)

    return {
        "total_words_with_confidence": total,
        "low_confidence_words_count": low_count,
        "very_low_confidence_words_count": very_low_count,
        "low_confidence_words_ratio": low_ratio,
        "low_confidence_important_words": low_confidence_important_words[:MAX_EXPORTED_CONFIDENCE_WORDS],
        "very_low_confidence_words": very_low_confidence_words[:MAX_EXPORTED_CONFIDENCE_WORDS],
        "removable_low_confidence_noise_words": removable_low_confidence_noise_words[:MAX_EXPORTED_CONFIDENCE_WORDS],
    }


def remove_low_confidence_noise_words(
    text: str,
    word_confidence_stats: dict[str, Any],
) -> tuple[str, list[str]]:
    if not REMOVE_LOW_CONFIDENCE_NOISE_WORDS:
        return text, []

    removed = []
    candidates = word_confidence_stats.get("removable_low_confidence_noise_words") or []

    for item in candidates:
        word = normalize_text(str(item.get("word", "")))

        if not word:
            continue

        pattern = rf"\b{re.escape(word)}\b"

        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
            removed.append("low_confidence_noise_word")

    text = re.sub(r"\s+", " ", text).strip()
    return text, sorted(set(removed))


# обезличивание

def mask_long_number_word_sequences(text: str, min_len: int = 4) -> tuple[str, list[str]]:
    tokens = text.split()
    new_tokens = []
    removed = []

    i = 0
    while i < len(tokens):
        if tokens[i] not in NUMBER_WORDS:
            new_tokens.append(tokens[i])
            i += 1
            continue

        j = i
        while j < len(tokens) and tokens[j] in NUMBER_WORDS:
            j += 1

        if j - i >= min_len:
            new_tokens.append("[NUMBER_SEQ]")
            removed.append("spoken_number_sequence")
        else:
            new_tokens.extend(tokens[i:j])

        i = j

    return " ".join(new_tokens), removed


def anonymize_pii(text: str) -> tuple[str, list[str]]:
    removed = []

    patterns = [
        (r"\b\d{12}\b", "[IIN]", "iin_digits"),
        (r"\b(?:\d[ -]?){16}\b", "[CARD]", "card_digits"),
        (r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}", "[PHONE]", "phone_digits"),
        (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL]", "email"),
        (r"\b[а-яё]{4,}(?:ова|ева|ина|ская|цкая)\b", "[NAME]", "single_surname_ru"),
        (r"\b[а-яәғқңөұүһі]+нің\s+[а-яәғқңөұүһі]+\s+[а-яәғқңөұүһі]+(?:қызы|ұлы)\b", "[NAME]", "name_kk"),
    ]

    for pattern, replacement, reason in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            removed.append(reason)

    text, spoken_number_removed = mask_long_number_word_sequences(text)
    removed.extend(spoken_number_removed)

    text = re.sub(r"\s+", " ", text).strip()
    return text, removed

# очистка сегментов

def is_only_empty_or_number_words(text: str) -> bool:
    text = normalize_text(text)
    words = text.split()

    if not words:
        return True

    allowed_words = EMPTY_PHRASES | NUMBER_WORDS

    return all(word in allowed_words or word.isdigit() for word in words)


def is_empty_or_noninformative(text: str) -> tuple[bool, str]:
    text = normalize_text(text)

    if not text:
        return True, "empty_text"

    if text in EMPTY_PHRASES:
        return True, "empty_phrase"

    if is_only_empty_or_number_words(text) and not has_problem_signal(text):
        return True, "only_empty_or_number_words"

    words_count = count_words(text)

    if words_count <= 3 and not has_problem_signal(text):
        return True, "too_short_without_problem_signal"

    if words_count <= VERY_SHORT_CALL_WORDS and not has_problem_signal(text):
        return True, "short_noninformative"

    return False, "useful"


def clean_segment_text(raw_text: str, word_confidence_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    text = normalize_text(raw_text)
    removed_parts = []

    text, removed = apply_patterns(text, BOILERPLATE_PATTERNS)
    removed_parts.extend(removed)

    text, removed = apply_patterns(text, ASR_GARBAGE_PATTERNS)
    removed_parts.extend(removed)

    if word_confidence_stats is not None:
        text, removed = remove_low_confidence_noise_words(text, word_confidence_stats)
        removed_parts.extend(removed)

    text, removed = apply_patterns(text, OPERATOR_PATTERNS)
    removed_parts.extend(removed)

    # После приветствий иногда остаётся "алло", "иә", "мхм" и т.п.
    text, removed = apply_patterns(text, CLOSING_PATTERNS)
    removed_parts.extend(removed)

    text, removed = apply_patterns(text, VERIFICATION_PATTERNS)
    removed_parts.extend(removed)

    text, removed = anonymize_pii(text)
    removed_parts.extend(removed)

    text = normalize_text(text)

    if "[name]" in text.lower() and not has_problem_signal(text):
        text = ""
        removed_parts.append("drop_pii_only_segment")

    is_empty, empty_reason = is_empty_or_noninformative(text)

    return {
        "cleaned_text": "" if is_empty else text,
        "is_empty_segment": is_empty,
        "empty_reason": empty_reason,
        "removed_parts": sorted(set(removed_parts)),
        "word_count_cleaned": 0 if is_empty else count_words(text),
    }


def clean_segments(
    segments: list[dict[str, Any]],
    word_confidence_stats: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept_segments = []
    removed_segments = []

    for index, segment in enumerate(segments):
        raw_text = segment.get("text", "")
        cleaned = clean_segment_text(raw_text, word_confidence_stats)

        segment_result = {
            "index": index,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "raw_text": raw_text,
            "cleaned_text": cleaned["cleaned_text"],
            "removed_parts": cleaned["removed_parts"],
            "empty_reason": cleaned["empty_reason"],
        }

        if cleaned["is_empty_segment"]:
            removed_segments.append(segment_result)
        else:
            kept_segments.append(segment_result)

    return kept_segments, removed_segments


def build_cleaned_transcript(
    item: dict[str, Any],
    word_confidence_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    segments = item.get("segments") or []

    if segments:
        kept_segments, removed_segments = clean_segments(segments, word_confidence_stats)
        cleaned_text = " ".join(
            segment["cleaned_text"]
            for segment in kept_segments
            if segment["cleaned_text"]
        )
    else:
        cleaned = clean_segment_text(item.get("text", ""), word_confidence_stats)
        kept_segments = []
        removed_segments = []
        cleaned_text = cleaned["cleaned_text"]

    cleaned_text = normalize_text(cleaned_text)
    is_empty_call, empty_reason = is_empty_or_noninformative(cleaned_text)

    return {
        "cleaned_transcript": "" if is_empty_call else cleaned_text,
        "is_empty_call": is_empty_call,
        "empty_reason": empty_reason,
        "segments_cleaned": kept_segments,
        "segments_removed": removed_segments,
        "word_count_cleaned": 0 if is_empty_call else count_words(cleaned_text),
    }


def extract_problem_keywords(text: str) -> list[str]:
    text = normalize_text(text)

    found = [
        keyword
        for keyword in PROBLEM_KEYWORDS
        if normalize_text(keyword) in text
    ]

    return sorted(set(found))



def get_language_code(item: dict[str, Any]) -> str | None:
    language = item.get("language")

    if isinstance(language, dict):
        code = language.get("code")
        return str(code) if code is not None else None

    if isinstance(language, str):
        return language

    selected_language = item.get("selected_language")
    if selected_language is not None:
        return str(selected_language)

    return None


def get_language_confidence(item: dict[str, Any]) -> str | None:
    language = item.get("language")

    if isinstance(language, dict):
        confidence = language.get("confidence")
        return str(confidence) if confidence is not None else None

    language_detection = item.get("language_detection")
    if isinstance(language_detection, dict):
        confidence = language_detection.get("confidence")
        return str(confidence) if confidence is not None else None

    return None


def get_raw_word_count(item: dict[str, Any]) -> int | None:
    word_count = item.get("word_count")

    if isinstance(word_count, int):
        return word_count

    words = item.get("words")
    if isinstance(words, list):
        return len(words)

    text = item.get("text")
    if isinstance(text, str):
        return count_words(text)

    return None


def get_average_word_confidence(item: dict[str, Any]) -> float | None:
    avg_confidence = item.get("avg_confidence")

    if isinstance(avg_confidence, (int, float)):
        return round(float(avg_confidence), 4)

    words = item.get("words") or []
    confidences = [
        float(word.get("conf"))
        for word in words
        if isinstance(word, dict) and isinstance(word.get("conf"), (int, float))
    ]

    if not confidences:
        return None

    return round(sum(confidences) / len(confidences), 4)

def build_quality_flags(
    item: dict[str, Any],
    cleaned: dict[str, Any],
    word_confidence_stats: dict[str, Any],
) -> list[str]:
    flags = []

    avg_confidence = item.get("avg_confidence")
    if isinstance(avg_confidence, (int, float)) and avg_confidence < LOW_CONFIDENCE_THRESHOLD:
        flags.append("low_avg_confidence")

    language_confidence = get_language_confidence(item)

    if language_confidence == "low":
        flags.append("low_language_confidence")

    if item.get("fallback_language_used"):
        flags.append("fallback_language_used")

    asr_noise_count = 0
    pii_removed = False

    pii_reasons = {
        "iin_digits",
        "card_digits",
        "phone_digits",
        "email",
        "name_ru",
        "name_kk",
        "single_surname_ru",
        "spoken_number_sequence",
        "drop_pii_only_segment",
    }

    for segment in cleaned["segments_cleaned"] + cleaned["segments_removed"]:
        for part in segment.get("removed_parts", []):
            if part.startswith("asr_"):
                asr_noise_count += 1

            if part in pii_reasons:
                pii_removed = True

    if asr_noise_count >= 2:
        flags.append("many_asr_noise_fragments")

    if pii_removed:
        flags.append("pii_removed")

    if "[number_seq]" in cleaned.get("cleaned_transcript", "").lower():
        flags.append("contains_number_sequence")

    cleaning_ratio = calculate_cleaning_ratio(
        item.get("word_count"),
        cleaned.get("word_count_cleaned"),
    )

    if (
        isinstance(cleaning_ratio, float)
        and not cleaned["is_empty_call"]
        and cleaning_ratio < HIGH_CLEANING_RATIO_THRESHOLD
    ):
        flags.append("high_cleaning_ratio")

    if cleaned["word_count_cleaned"] <= VERY_SHORT_CALL_WORDS and not cleaned["is_empty_call"]:
        flags.append("very_short_but_useful")

    low_ratio = word_confidence_stats.get("low_confidence_words_ratio")
    if isinstance(low_ratio, float) and low_ratio >= HIGH_LOW_CONFIDENCE_WORD_RATIO:
        flags.append("many_low_confidence_words")

    if word_confidence_stats.get("very_low_confidence_words_count", 0) > 0:
        flags.append("has_very_low_confidence_words")

    if word_confidence_stats.get("low_confidence_important_words"):
        flags.append("low_confidence_important_words")

    if word_confidence_stats.get("removable_low_confidence_noise_words"):
        flags.append("has_low_confidence_noise_candidates")

    if word_confidence_stats.get("removed_low_confidence_words"):
        flags.append("removed_low_confidence_noise_words")

    return sorted(set(flags))


def count_quality_flags(results: list[dict[str, Any]]) -> dict[str, int]:
    flag_counts = {}

    for item in results:
        for flag in item.get("quality_flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return dict(sorted(flag_counts.items()))

# итоговая оценка для ревью

def build_review_priority(
    is_empty_call: bool,
    cleaned_transcript: str,
    quality_flags: list[str],
) -> str:
    if is_empty_call:
        return "high"

    if not cleaned_transcript:
        return "high"

    flags = set(quality_flags)

    high_flags = {
        "transcription_error",
        "low_avg_confidence",
        "low_language_confidence",
        "fallback_language_used",
        "many_asr_noise_fragments",
        "high_cleaning_ratio",
    }

    medium_flags = {
        "has_very_low_confidence_words",
        "has_low_confidence_noise_candidates",
        "contains_number_sequence",
        "pii_removed",
        "very_short_but_useful",
    }

    if flags & high_flags:
        return "high"

    if flags & medium_flags:
        return "medium"

    return "low"


def build_ready_for_classification(
    is_empty_call: bool,
    cleaned_transcript: str,
    review_priority: str,
) -> bool:
    if is_empty_call:
        return False

    if not cleaned_transcript:
        return False

    if review_priority == "high":
        return False

    return True


def count_review_priorities(results: list[dict[str, Any]]) -> dict[str, int]:
    priority_counts = {
        "low": 0,
        "medium": 0,
        "high": 0,
    }

    for item in results:
        priority = item.get("review_priority", "high")

        if priority not in priority_counts:
            priority = "high"

        priority_counts[priority] += 1

    return priority_counts


def process_item(item: dict[str, Any]) -> dict[str, Any]:
    selected_language = get_language_code(item)
    language_confidence = get_language_confidence(item)
    raw_word_count = get_raw_word_count(item)
    avg_confidence = get_average_word_confidence(item)

    if "error" in item:
        return {
            "file_name": item.get("file_name"),
            "file_path": item.get("file_path"),
            "status": "error",
            "error": item.get("error"),
            "raw_transcript": "",
            "cleaned_transcript": "",
            "selected_language": selected_language,
            "language_confidence": language_confidence,
            "is_empty_call": True,
            "empty_reason": "transcription_error",
            "quality_flags": ["transcription_error"],
            "word_confidence_stats": {},
            "review_priority": "high",
            "ready_for_classification": False,
        }

    word_confidence_stats = analyze_word_confidences(item)
    cleaned = build_cleaned_transcript(item, word_confidence_stats)
    cleaned_transcript = cleaned["cleaned_transcript"]

    removed_parts = []
    for segment in cleaned["segments_cleaned"] + cleaned["segments_removed"]:
        removed_parts.extend(segment.get("removed_parts", []))

    if "low_confidence_noise_word" in removed_parts:
        word_confidence_stats["removed_low_confidence_words"] = word_confidence_stats.get(
            "removable_low_confidence_noise_words",
            [],
        )
    else:
        word_confidence_stats["removed_low_confidence_words"] = []

    quality_flags = build_quality_flags(item, cleaned, word_confidence_stats)

    review_priority = build_review_priority(
        is_empty_call=cleaned["is_empty_call"],
        cleaned_transcript=cleaned_transcript,
        quality_flags=quality_flags,
    )

    ready_for_classification = build_ready_for_classification(
        is_empty_call=cleaned["is_empty_call"],
        cleaned_transcript=cleaned_transcript,
        review_priority=review_priority,
    )

    cleaning_ratio = calculate_cleaning_ratio(
        raw_word_count,
        cleaned["word_count_cleaned"],
    )

    return {
        "file_name": item.get("file_name"),
        "file_path": item.get("file_path"),
        "status": "processed",
        "selected_language": selected_language,
        "language_confidence": language_confidence,

        "raw_transcript": item.get("text", ""),
        "cleaned_transcript": cleaned_transcript,

        "is_empty_call": cleaned["is_empty_call"],
        "empty_reason": cleaned["empty_reason"],

        "review_priority": review_priority,
        "ready_for_classification": ready_for_classification,

        "word_count_raw": raw_word_count,
        "word_count_cleaned": cleaned["word_count_cleaned"],
        "cleaning_ratio": cleaning_ratio,

        "avg_confidence": avg_confidence,
        "transcription_time_seconds": item.get("transcription_time_seconds"),
        "total_time_with_detection_seconds": item.get("total_time_with_detection_seconds"),

        "problem_keywords_found": extract_problem_keywords(cleaned_transcript),
        "removed_parts": sorted(set(removed_parts)),
        "quality_flags": quality_flags,
        "word_confidence_stats": word_confidence_stats,

        "segments_cleaned": cleaned["segments_cleaned"],
        "segments_removed": cleaned["segments_removed"],
    }

def save_review_csv(results: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "file_name",
        "selected_language",
        "language_confidence",
        "is_empty_call",
        "empty_reason",
        "review_priority",
        "ready_for_classification",
        "quality_flags",
        "word_count_raw",
        "word_count_cleaned",
        "cleaning_ratio",
        "avg_confidence",
        "total_words_with_confidence",
        "low_confidence_words_count",
        "very_low_confidence_words_count",
        "low_confidence_words_ratio",
        "low_confidence_important_words",
        "very_low_confidence_words",
        "candidate_low_confidence_noise_words",
        "removed_low_confidence_words",
        "problem_keywords_found",
        "removed_parts",
        "raw_transcript",
        "cleaned_transcript",
    ]

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for item in results:
            stats = item.get("word_confidence_stats", {}) or {}

            writer.writerow({
                "file_name": item.get("file_name"),
                "selected_language": item.get("selected_language"),
                "language_confidence": item.get("language_confidence"),
                "is_empty_call": item.get("is_empty_call"),
                "empty_reason": item.get("empty_reason"),
                "review_priority": item.get("review_priority"),
                "ready_for_classification": item.get("ready_for_classification"),
                "quality_flags": ", ".join(item.get("quality_flags", [])),
                "word_count_raw": item.get("word_count_raw"),
                "word_count_cleaned": item.get("word_count_cleaned"),
                "cleaning_ratio": item.get("cleaning_ratio"),
                "avg_confidence": item.get("avg_confidence"),
                "total_words_with_confidence": stats.get("total_words_with_confidence"),
                "low_confidence_words_count": stats.get("low_confidence_words_count"),
                "very_low_confidence_words_count": stats.get("very_low_confidence_words_count"),
                "low_confidence_words_ratio": stats.get("low_confidence_words_ratio"),
                "low_confidence_important_words": format_confidence_words(stats.get("low_confidence_important_words", [])),
                "very_low_confidence_words": format_confidence_words(stats.get("very_low_confidence_words", [])),
                "candidate_low_confidence_noise_words": format_confidence_words(stats.get("removable_low_confidence_noise_words", [])),
                "removed_low_confidence_words": format_confidence_words(stats.get("removed_low_confidence_words", [])),
                "problem_keywords_found": ", ".join(item.get("problem_keywords_found", [])),
                "removed_parts": ", ".join(item.get("removed_parts", [])),
                "raw_transcript": item.get("raw_transcript", ""),
                "cleaned_transcript": item.get("cleaned_transcript", ""),
            })

def build_summary(processed_results: list[dict[str, Any]]) -> dict[str, Any]:
    total_count = len(processed_results)
    empty_count = sum(1 for item in processed_results if item.get("is_empty_call"))
    useful_count = total_count - empty_count
    error_count = sum(1 for item in processed_results if item.get("status") == "error")
    review_needed_count = sum(1 for item in processed_results if item.get("quality_flags"))
    quality_flags_counts = count_quality_flags(processed_results)
    review_priority_counts = count_review_priorities(processed_results)
    ready_for_classification_count = sum(
        1 for item in processed_results
        if item.get("ready_for_classification")
    )
    not_ready_for_classification_count = total_count - ready_for_classification_count

    return {
        "total_count": total_count,
        "useful_calls_count": useful_count,
        "empty_calls_count": empty_count,
        "error_count": error_count,
        "review_needed_count": review_needed_count,
        "quality_flags_counts": quality_flags_counts,
        "review_priority_counts": review_priority_counts,
        "ready_for_classification_count": ready_for_classification_count,
        "not_ready_for_classification_count": not_ready_for_classification_count,
    }


def build_many_output(input_json: Path, processed_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_file": str(input_json),
        "preprocessing_type": "rule_based_minimal_v6_dual_input",
        "input_mode": "many_transcripts",
        "notes": [
            "raw_transcript не перезаписывается",
            "cleaned_transcript собирается из очищенных сегментов с таймингами",
            "очистка консервативная: удаляется технический мусор, но смысловые проблемы остаются",
            "word-level confidence используется для quality flags и очень осторожного удаления низкоуверенного мусора",
        ],
        "summary": build_summary(processed_results),
        "results": processed_results,
    }


def preprocess_many_transcripts_file(
    input_json: Path,
    output_json: Path,
    review_csv: Path | None = None,
) -> dict[str, Any]:
    source_data = load_json(input_json)

    if not isinstance(source_data, dict):
        raise ValueError("JSON с несколькими транскриптами должен быть объектом с ключом 'results'.")

    source_results = source_data.get("results")

    if not isinstance(source_results, list):
        raise ValueError("Для режима many нужен JSON формата: {'results': [...]}.")

    processed_results = [process_item(item) for item in source_results]
    output_data = build_many_output(input_json, processed_results)

    save_json(output_data, output_json)

    if review_csv is not None:
        save_review_csv(processed_results, review_csv)

    return output_data


def preprocess_single_transcript_file(
    input_json: Path,
    output_json: Path,
    review_csv: Path | None = None,
) -> dict[str, Any]:
    source_data = load_json(input_json)

    if not isinstance(source_data, dict):
        raise ValueError("JSON с одним транскриптом должен быть объектом, а не списком.")

    if isinstance(source_data.get("results"), list):
        raise ValueError("Для режима single нужен JSON одного транскрипта, без ключа 'results'.")

    processed_item = process_item(source_data)
    output_data = {
        "source_file": str(input_json),
        "preprocessing_type": "rule_based_minimal_v6_dual_input",
        "input_mode": "single_transcript",
        "result": processed_item,
    }

    save_json(output_data, output_json)

    if review_csv is not None:
        save_review_csv([processed_item], review_csv)

    return output_data


def preprocess_file(
    input_json: Path,
    output_json: Path,
    review_csv: Path | None = None,
    mode: str = "auto",
) -> dict[str, Any]:
    if mode not in {"auto", "many", "single"}:
        raise ValueError("mode должен быть одним из: auto, many, single")

    if mode == "many":
        return preprocess_many_transcripts_file(
            input_json=input_json,
            output_json=output_json,
            review_csv=review_csv,
        )

    if mode == "single":
        return preprocess_single_transcript_file(
            input_json=input_json,
            output_json=output_json,
            review_csv=review_csv,
        )

    source_data = load_json(input_json)

    if isinstance(source_data, dict) and isinstance(source_data.get("results"), list):
        processed_results = [process_item(item) for item in source_data["results"]]
        output_data = build_many_output(input_json, processed_results)
        save_json(output_data, output_json)

        if review_csv is not None:
            save_review_csv(processed_results, review_csv)

        return output_data

    if isinstance(source_data, dict):
        processed_item = process_item(source_data)
        output_data = {
            "source_file": str(input_json),
            "preprocessing_type": "rule_based_minimal_v6_dual_input",
            "input_mode": "single_transcript",
            "result": processed_item,
        }
        save_json(output_data, output_json)

        if review_csv is not None:
            save_review_csv([processed_item], review_csv)

        return output_data

    raise ValueError("Неизвестный формат JSON: ожидается объект одного транскрипта или объект с ключом 'results'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="rule-based preprocessing"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help=f"default path: {DEFAULT_INPUT_JSON}",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"default path: {DEFAULT_OUTPUT_JSON}",
    )

    parser.add_argument(
        "--review-csv",
        type=Path,
        default=DEFAULT_REVIEW_CSV,
        help=f"csv path: {DEFAULT_REVIEW_CSV}",
    )

    parser.add_argument(
        "--no-review-csv",
        action="store_true",
        help="do not save CSV",
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "many", "single"],
        default="auto",
        help="auto определяет формат сам; many ждёт {'results': [...]}; single ждёт один транскрипт",
    )

    return parser.parse_args()


def print_summary(output_data: dict[str, Any], output_json: Path, review_csv: Path | None) -> None:
    print(f"JSON: {output_json}")

    if review_csv is not None:
        print(f"CSV: {review_csv}")

    input_mode = output_data.get("input_mode")
    print(f"Mode: {input_mode}")

    if input_mode == "single_transcript":
        result = output_data.get("result", {})
        print(f"File: {result.get('file_name')}")
        print(f"Language: {result.get('selected_language')} | confidence: {result.get('language_confidence')}")
        print(f"Raw words: {result.get('word_count_raw')}")
        print(f"Cleaned words: {result.get('word_count_cleaned')}")
        print(f"Review priority: {result.get('review_priority')}")
        print(f"Ready for classification: {result.get('ready_for_classification')}")
        return

    summary = output_data["summary"]

    print(f"Total calls: {summary['total_count']}")
    print(f"Useful calls: {summary['useful_calls_count']}")
    print(f"Empty/ garbage calls: {summary['empty_calls_count']}")
    print(f"Errors: {summary['error_count']}")
    print(f"Review needed: {summary['review_needed_count']}")
    print(f"Quality flags: {summary['quality_flags_counts']}")
    print(f"Review priority: {summary['review_priority_counts']}")
    print(f"Ready for classification: {summary['ready_for_classification_count']}")
    print(f"Not ready for classification: {summary['not_ready_for_classification_count']}")


def main() -> None:
    args = parse_args()

    review_csv = None if args.no_review_csv else args.review_csv

    output_data = preprocess_file(
        input_json=args.input,
        output_json=args.output_json,
        review_csv=review_csv,
        mode=args.mode,
    )

    print_summary(
        output_data=output_data,
        output_json=args.output_json,
        review_csv=review_csv,
    )

if __name__ == "__main__":
    main()
