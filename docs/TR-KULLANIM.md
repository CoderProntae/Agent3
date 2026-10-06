# Agent3 — Türkçe Kullanım Kılavuzu

**Agent3**, kendi bilgisayarınızda çalışan otonom bir yapay zekâ kodlama ajanıdır. Masaüstü
uygulaması (PySide6/Qt6) içinde; sohbet paneli, dosya gezgini, kod düzenleyici, canlı fark (diff)
görüntüleyici, gömülü terminal ve git entegrasyonu bulunur. Model tamamen yereldeki **Ollama**
sunucusundan (`http://localhost:11435`) çalışır — hiçbir veri buluta gitmez.

---

## 1. Kurulum

### A) Hazır `.exe` ile (önerilen)

1. GitHub deposunda **Actions → Build & Release → Run workflow**'a basın
   (sürüm taslağı istiyorsanız *create_release* kutusunu işaretleyin). Alternatif olarak
   `v1.0.0` gibi bir etiket (tag) gönderin.
2. İş akışı bitince **`Agent3-windows-x64-bundle`** çıktısını indirip açın. İçinde:
   - `Agent3.exe` — tek dosyalık ana uygulama (kurulum gerekmez)
   - `INSTALL.txt`, `README.md`, `LICENSE`
3. [Ollama](https://ollama.com) kurun ve **11435 portunda** başlatın:

   ```bat
   set OLLAMA_HOST=127.0.0.1:11435
   ollama serve
   ```

4. Bir model indirin (varsayılan etiket `qwen3.5-9b-abliterated`, kurulu olan herhangi bir model olur):

   ```bat
   ollama pull qwen2.5-coder:7b
   ```

5. `Agent3.exe`'yi çalıştırın.

### B) Kaynaktan çalıştırma

```bash
python -m venv .venv
.venv\Scripts\activate         # Linux/macOS: . .venv/bin/activate
pip install -r requirements-dev.txt
python -m agent3               # ana uygulama
python -m agent3 --self-test   # model olmadan kurulumu doğrular
```

---

## 2. İlk kullanım

1. **Ctrl+O** ile bir çalışma klasörü seçin. Ajan **yalnızca** bu klasörün içine erişebilir.
2. Araç çubuğundaki bağlantı rozetine bakın: yeşil = Ollama'ya bağlanıldı, kırmızı = bağlanılamadı.
   Kırmızıysa **File → Settings → Connection → Test connection** ile deneyin.
3. Model seçiciden kurulu modellerden birini seçin.
4. Alttaki kutuya görevi yazın ve **Ctrl+Enter**'a basın. Örnek:

   > "FastAPI ile bir /health uç noktası ekle, pytest testini yaz ve testleri çalıştır."

5. Ajan çalışırken orta panelde **eylem kartları** görünür
   (`[AGENT] write_file … ✓ 12 ms`). Her kartın "Details" düğmesiyle çıktıyı/diff'i açabilirsiniz.
6. Sağ panelde değişikliklerin renkli farkı, alt panelde komut çıktıları canlı akar.
7. Durdurmak için **Esc** veya **Stop** düğmesi.

---

## 3. Arayüz rehberi

| Bölge | İçerik |
|---|---|
| Üst araç çubuğu | Çalışma klasörü ve menüler |
| Sol kenar çubuğu | Dosya gezgini · **ajan planı (PLAN)** · oturum listesi |
| Orta panel | Sohbet (tam markdown desteği, aşağıda) ve canlı eylem kartları |
| **Mesaj kutusunun altındaki şerit** | **Model seçici · düşünme anahtarı · düşünme düzeyi · bağlantı durumu** |
| Sağ panel | Sekmeli kod düzenleyici + satır içi / yan yana fark görüntüleyici |
| Alt panel | Gömülü terminal; başlıkta `● N background` rozeti arka planda çalışan süreçleri gösterir |
| Durum çubuğu | Ajan durumu ve geçici bildirimler (klasör bağlandı, plan güncellendi…) |

### Sohbette markdown

Qt'nin kendi `setMarkdown` fonksiyonu belgeyi programatik kuruyor ve tema
stilini hiç okumuyordu; bu yüzden renkler, çerçeveler ve arka planlar çöpe
gidiyor, cevaplar tek düze gri metin olarak görünüyordu. Artık kendi HTML
üreticimiz var (`agent3/ui/markdown_render.py`):

| Markdown | Nasıl görünüyor |
|---|---|
| Başlıklar | Kademeli punto, `#` ve `##` altında ince çizgi |
| `satır içi kod` | Arka plan tonu, tek aralıklı yazı tipi |
| ``` kod blokları | **Sözdizimi renklendirmesi** (Python, JS/TS, JSON, YAML, TOML, shell, C ailesi, SQL, CSS, HTML, diff), dil etiketi ve yana taşmak yerine satır sarma |
| Tablolar | Gölgeli başlık satırı, gerçek kenarlıklar, `:---:` hizalama |
| `- [x]` / `- [ ]` | ✓ / ○ kutucukları — ajanın planı plan gibi okunuyor |
| Listeler | Girintiye göre iç içe, renkli işaretler |
| Alıntılar | Solda vurgu çubuğu, soluk metin |
| Bağlantılar ve URL'ler | Tıklanabilir, vurgu renginde |

Her şey önce HTML olarak kaçışlanır; kod bloğunun veya `ters tırnağın` içindeki
markdown olduğu gibi kalır.

---

### Model ve düşünme şeridi

Model seçici artık üst çubukta değil, **yazdığınız kutunun hemen altında** —
çünkü her mesajda değiştirebileceğiniz ayarlar oraya aittir.

| Kontrol | Ne yapar |
|---|---|
| `◆ model` | Sunucuda kurulu modeller; elle de yazabilirsiniz |
| `Thinking` kutusu | Modelin yanıtlamadan önce akıl yürütmesini açar/kapatır |
| `effort` listesi | Düşünme düzeyi |
| `● online · …` | Ollama bağlantı durumu (üstüne gelin: sürüm ve adres) |

**Önemli: sınıflandırma model ADINA göre değil, kanıta göre yapılır.**
Agent3'te "şu aile şunu destekler" diye bir tablo yoktur. Model değiştiğinde
`/api/show` sorulur ve şu sırayla gerçek kanıt aranır:

| # | Kaynak | Ne kanıtlar |
|---|---|---|
| 1 | `thinking: {"values": [...], "default": ...}` | Kesin bilgi; sunucu `think` alanını kendisi uygular |
| 2 | Modelin **chat template**'i | `enable_thinking` → aç/kapa; `reasoning_effort`'ün karşılaştırıldığı tırnaklı liste → **gerçek seviye adları** |
| 3 | `capabilities` içinde `"thinking"` | Yalnızca aç/kapa, seviye yok |
| 4 | Hiçbiri | Model akıl yürütmüyor |

Şeritte **kaynağın adı değil, seçeneğin kendisi** yazar: modelin kabul ettiği
seviyelerden oluşan bir liste. Liste, anahtar kapalıyken bile tıklanabilir —
bir seviye seçmek düşünmeyi açar. Bilginin nereden geldiği etiketin değil,
**üstüne gelince çıkan ipucunun** içindedir.

2. adım, topluluk GGUF paketlerini çalıştıran şeydir. Qwen3.x template'i şunu
içerir:

```jinja
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
```

Bu yüzden `bernquant/Qwen3.5-9B-…-GGUF` gibi bir modelde **X-High / Medium /
Low** seçenekleri çıkar (varsayılan `xhigh`), adı ne olursa olsun.

#### "Kapattım ama yine düşünüyor" sorunu

Ollama `think` alanını **her** chat template'ine iletmez. Template
`{%- if enable_thinking is undefined or enable_thinking is true %}` ile
başlıyorsa, "bir şey söylenmedi" durumunu "tam efor ile düşün" sayar — kapalı
olmasına rağmen düşünmesinin sebebi tam olarak budur. Sunucu yerel
`thinking` verisi bildirmediğinde iki savunma birlikte devreye girer:

* **Ayar istemin içinde tekrarlanır.** Seviye için template'in *kendi* cümlesi
  (`"Reasoning effort is set to low. Keep your thinking brief and focused…"`)
  ilk sistem mesajının önüne — template'in basacağı yere — eklenir. "Kapalı"
  için belgelenmiş `/no_think` yumuşak anahtarı son kullanıcı mesajına eklenir
  ve açık bir talimat yazılır. Hiçbir metin uydurulmaz; cümleler template'in
  içinden birebir alınır.
* **Satır içi düşünme süzülür.** Model yine de düşünür ve sunucu izi
  `message.content` içinde döndürürse, akış sırasında çalışan bir durum
  makinesi `<think>`, `<thinking>`, `<reasoning>`, `◁think▷` bloklarını
  yanıttan ayırır (parça sınırında bölünmüş etiketleri de yakalar) ve
  Reasoning kanalına yollar. Böylece araç ayrıştırıcısı modelin müsveddesini
  asla görmez; durum çubuğunda bir kez "bu model kapalıyken de düşünüyor"
  uyarısı belirir.

Her ikisi de **Ayarlar → Transport** altından kapatılabilir. Modelin kabul
etmediği bir değer isteğe hiç konmaz, böylece sunucu isteği tümden reddetmez.
Akıl yürütme metni yanıttan ayrı bir **"Reasoning"** bloğunda akar; blok
varsayılan olarak kapalıdır, tıklayarak açarsınız.

### Neden bazen geç yanıt veriyordu?

Yerel sunucu KV önbelleğini yalnızca iki istemin **ortak ön ekine** uygular.
6 GB'lık bir kartta 9B model, istem işlemeyi saniyede ~50 token hızında yapar;
yani ön eki kaybetmek dakikalara mal olur. Bu yüzden bağlam düzeni Agent3'te
bir performans sözleşmesidir. Üç kural:

1. **0 numaralı mesaj asla değişmez.** Sistem istemi kimlik, kurallar, araç
   kataloğu ve çıktı sözleşmesinden ibarettir: her adımda, her turda ve her
   çalışma klasöründe bayt bayt aynı olan ~2500 token.
2. **Değişken durum araya yazılmaz, sona eklenir.** Çalışma alanı fotoğrafı
   (ağaç, git durumu, tarih) artık her turun kullanıcı mesajının başında bir
   **sohbet turu** olarak gider. Böylece yeni bir ağaç, ağacın token'ı kadar
   maliyetlidir — tüm konuşmanın token'ı kadar değil.
3. **Bağlam penceresinin sol kenarı sabittir.** Akla ilk gelen sınır olan
   `history[-40:]`, konuşma uzadığında her adımda en eski mesajı atar; her
   atış geri kalanı kaydırdığı için ön ek **her adımda** çöper. Agent3'ün
   penceresi bunun yerine bütçesine kadar büyür, sonra %60'ına geri düşer:
   pahalı adım her mesajda değil, ~16 mesajda bir yaşanır.

Her iki hata da gerçek bir llama.cpp logunda bulundu:

```
slot get_availabl: - checking sim = 0.177 (2524/14252)
slot print_timing: prompt processing ... 50 tokens per second
```

Eşleşen 2524 token tam olarak sabit ön ekti; kalan 11 728 token yeniden
hesaplandı — yanıtın ilk token'ından önce yaklaşık **dört dakika**. 40
mesajlık pencereyle 120 mesajlık bir konuşmada iki düzeltme birlikte yeniden
işlenen token'ı **%93** azaltıyor (905 614 → 65 436).

Ortam satırında ayrıca yalnızca tarih vardır; her dakika değişen bir saat tek
başına önbelleği geçersiz kılardı.

Düşünme izi gizliyken durum çubuğu karakter sayısını yazar; böylece bir dakika
düşünen bir model asla "takılmış" gibi görünmez.

### PLAN paneli

Ajan iki adımdan uzun işlerde önce planını yazar (`manage_tasks` aracı).
Sol kenar çubuğundaki PLAN bölümü bunu canlı gösterir: `✓` biten,
`◐` üzerinde çalışılan, `○` bekleyen adım; üstte `1/4` sayacı ve ilerleme
çubuğu. Planda açık madde kaldığı sürece ajanın "bitirdim" demesi bir kez
reddedilir.

### Arka plan süreçleri

`npm run dev`, `uvicorn`, `watch` gibi bitmeyen komutlar `run_command` ile
değil `start_process` ile çalıştırılır; ajan beklemeden işine devam eder,
çıktısını `get_process_logs` ile okur, işi bitince `stop_process` ile kapatır.
Terminal başlığındaki `● N background` rozeti kaç sürecin ayakta olduğunu
söyler. Uygulamayı kapattığınızda bu süreçler otomatik sonlandırılır.

### Yazılan her dosya denetlenir

Ajan bir dosya yazdığı anda dosya ayrıştırılır (Python, JSON, TOML, YAML, XML,
INI, JavaScript, TypeScript; kuruluysa ayrıca `eslint` / `tsc` / `ruff`).
Sonuç araç çıktısına eklenir:

* `… [syntax OK (python-ast)]` → temiz;
* `SYNTAX ERROR x1 … line 42` → araç çağrısı **başarısız** sayılır ve ajan o
  adımda düzeltmek zorundadır; bozuk dosya varken `finish` reddedilir.

Bir düzenleme kötü gittiyse ajan `undo_file_change` ile dosyayı tek hamlede
önceki hâline döndürür (yeni oluşturulmuş bir dosyaysa siler).

**Kısayollar:** `Ctrl+O` klasör aç · `Ctrl+Enter` çalıştır · `Esc` durdur · `Ctrl+S` kaydet ·
`Ctrl+N` yeni oturum · `Ctrl+,` ayarlar · ``Ctrl+` `` terminali aç/kapat.

### Eylem kartlarını okumak

Ajanın her adımı sohbette bir kart olarak belirir:

| Öğe | Anlamı |
|---|---|
| `✓` yeşil | Araç başarılı |
| `✕` kırmızı | Araç hata verdi — ajan hatayı okuyup kendini düzeltir |
| `!` kırmızı | Komut güvenlik politikası veya interaktif komut tuzağı tarafından engellendi |
| `●` sarı | Hâlâ çalışıyor |
| `+8 -2` rozeti | Dosya değişikliğinde eklenen/silinen satır sayısı |
| **Diff** düğmesi | Kartı açar: **yalnızca değişen satırlar**, eski/yeni satır numaralarıyla, eklemeler yeşil (`+`), silmeler kırmızı (`-`), bağlam satırları soluk |
| **Details** düğmesi | Komut çıktısı: hatalar kırmızı, geçen testler yeşil, uyarılar sarı renklendirilir |

---

## 3.1 Ajanın çalışma disiplini

**Ucuz keşif.** Ajan büyük bir dosyayı okumadan önce `view_outline` çağırır:
sınıflar, fonksiyonlar, imzalar, docstring'ler ve satır numaraları — dosyanın
gövdesini bağlama yüklemeden. Sonra yalnızca ihtiyacı olan satır aralığını okur.

**Tek seferde düzenleme.** Bir dosyanın üç ayrı yeri değişecekse ajan üç kez
`edit_file` çağırmaz; tek bir `patch_file` ile çok parçalı (multi-hunk) unified
diff uygular. Parçalar satır numarasına değil **bağlama** göre yerleştirilir,
böylece önceki parça satır sayısını kaydırsa bile sonrakiler doğru yere oturur.

**Hiçbir komut sizi beklemez.** Ajanın içinde soruya cevap verebilecek kimse
yoktur. İki katman bunu engeller:

1. `npm init`, `apt-get install`, `-m`'siz `git commit`, `vim`, `less`, argümansız
   `python` gibi ~15 kalıp **çalıştırılmadan önce** reddedilir ve modele
   non-interaktif biçimi söylenir (`npm init -y` gibi).
2. Bir komut soru sorup susarsa (15 sn) süreç ağacı öldürülür. Çıktı satır
   satır değil ham blok olarak okunduğu için `Devam? [e/H] ` gibi **satır sonu
   olmayan** promptlar da görülür. Sessizce derleme yapan bir komut asla prompt
   sanılmaz.

**Bitmiş iş tanımı.** Ajan dosya değiştirdiyse ve son düzenlemeden sonra
başarılı bir komut çalıştırmadıysa `finish` **bir kez reddedilir** ve ajan
testlerine geri gönderilir (`pytest -q`, `npm test`, `go test ./...` veya bir
derleme/import kontrolü). Doğrulama başarısızsa farklı bir uyarı alır. Israr
ederse kilitlenme olmaz; ikinci deneme kabul edilir ama özetin altına görünür
bir uyarı düşülür. Kapatmak için `config.json` içinde
`agent.require_verification = false`.

---

## 4. Yol ve komut zekâsı

Küçük yerel modeller kabukta hep aynı dört hatayı yapar. Agent3 bunları artık kendi başına
yakalıyor, böylece ajan aynı yanlış yolu on kez denemiyor.

| Durum | Ne oluyor |
|---|---|
| Windows'ta `mv`, `cp`, `rm`, `ls`, `cat`, `touch`, `grep`, `sed` | Komut **çalıştırılmadan** reddedilir; yerine kullanılacak araç söylenir (`rename_file`, `delete_file`, `list_files`, `read_file`, `search_code`, …) |
| `cd olmayan-klasor & python app.py` | Çalıştırılmadan reddedilir. `cd` başarısız olsa bile Windows'ta `&` bir sonraki komutu yine çalıştırdığı için bu sessiz bir hatadır; doğru kullanım `cwd` argümanıdır |
| Komut hata verdi ve içinde olmayan bir yol var | Hata mesajına çalışma klasöründeki **en yakın gerçek yol** eklenir |
| Yolda boşluk var (`projede Duz/snake_game.py`) | Tırnak içine alınması gerektiği açıkça hatırlatılır |

Örnek: ajan `python projede Duz/snake_game.py` yazarsa, kabuk bunu iki ayrı argümana böler ve
komut başarısız olur. Agent3 çıktıya şunu ekler:

```
Komuttaki yollar bu çalışma klasörüyle eşleşmiyor:
- "Duz/snake_game.py" yok. Gerçek yol "projede Duz/snake_game.py" — içinde boşluk
  olduğu için komutta tırnak içine alınmalı.
```

Olmayan yollar yalnızca komut **başarısız olduktan sonra** bildirilir: `mkdir`, `git clone` veya bir
derleyici çıktısı zaten var olmayan bir yol yazmak zorundadır, bunları baştan engellemek yanlış olurdu.

---

## 5. Güvenlik

- Modelin ürettiği her yol `WorkspaceFS.resolve()` süzgecinden geçer: `..` ile yukarı çıkma,
  klasör dışı mutlak yollar ve sembolik bağlantı kaçışları reddedilir.
- Tehlikeli komutlar (`rm -rf /`, `mkfs`, `format C:`, fork bomb, `curl … | sh` …) engellenir;
  bu desenleri ayarlardan düzenleyebilirsiniz.
- Komutlar zaman aşımına uğrar ve durdurulduğunda tüm alt süreç ağacı sonlandırılır.
- GitHub kişisel erişim jetonu şifreli olarak saklanır, yalnızca git işlemlerinde kullanılır.

---

## 6. Veriler nerede saklanıyor?

| İşletim sistemi | Klasör |
|---|---|
| Windows | `%APPDATA%\Agent3` |
| macOS | `~/Library/Application Support/Agent3` |
| Linux | `~/.config/agent3` |

```
config.json          ayarlar (uç nokta, model, ajan davranışı, pencere durumu)
credentials.enc      şifreli GitHub jetonu
sessions.sqlite3     sohbet geçmişi
logs/agent3.log      döngüsel günlük dosyası
crashes/             beklenmeyen hata dökümleri
```

`AGENT3_HOME` ortam değişkeniyle bu klasörü taşıyabilirsiniz (taşınabilir kurulum).

---

## 7. Sık karşılaşılan sorunlar

| Belirti | Çözüm |
|---|---|
| Bağlantı rozeti kırmızı | `set OLLAMA_HOST=127.0.0.1:11435 && ollama serve` ile sunucuyu başlatın |
| "Model is not installed" uyarısı | `ollama pull <model>` ile indirin veya listeden kurulu bir model seçin |
| Araçlar "security violation" döndürüyor | Model çalışma klasörü dışına yazmaya çalıştı — bu bilinçli bir korumadır |
| Komut 124 koduyla bitti | Zaman aşımı; **Settings → Agent → Command timeout** değerini artırın |
| Git işlemleri çalışmıyor | `git` PATH'te kurulu olmalı |

---

## 8. Geliştirici notları

```bash
pytest -q                                    # tüm testler (160+)
QT_QPA_PLATFORM=offscreen pytest -q -m gui   # yalnızca arayüz testleri
python packaging/build.py --clean            # iki .exe'yi yerelde üret
```

Mimarinin ayrıntısı ve araç listesi için ana [`README.md`](../README.md) dosyasına bakın.
