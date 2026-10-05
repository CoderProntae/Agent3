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
   - `Agent3.exe` — ana uygulama
   - `UsageLimitEditor.exe` — yönetici kota aracı
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
python -m usage_limit_editor   # yönetici aracı
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
| Sol kenar çubuğu | Dosya gezgini · **ajan planı (PLAN)** · oturum listesi · kullanım/kota göstergeleri |
| Orta panel | Sohbet (markdown + kod vurgulama) ve canlı eylem kartları |
| **Mesaj kutusunun altındaki şerit** | **Model seçici · düşünme anahtarı · düşünme düzeyi · bağlantı durumu** |
| Sağ panel | Sekmeli kod düzenleyici + satır içi / yan yana fark görüntüleyici |
| Alt panel | Gömülü terminal; başlıkta `● N background` rozeti arka planda çalışan süreçleri gösterir |
| Durum çubuğu | Ajan durumu · bugünkü token · istek kotası |

### Model ve düşünme şeridi

Model seçici artık üst çubukta değil, **yazdığınız kutunun hemen altında** —
çünkü her mesajda değiştirebileceğiniz ayarlar oraya aittir.

| Kontrol | Ne yapar |
|---|---|
| `◆ model` | Sunucuda kurulu modeller; elle de yazabilirsiniz |
| `Thinking` kutusu | Modelin yanıtlamadan önce akıl yürütmesini açar/kapatır |
| `effort` listesi | Düşünme düzeyi |
| `● online · …` | Ollama bağlantı durumu (üstüne gelin: sürüm ve adres) |

**Önemli:** bu seçeneklerin içeriği uydurulmaz. Agent3 model değiştiğinde
Ollama'ya `/api/show` sorar ve modelin bildirdiği `thinking.values` listesini
aynen gösterir:

* `qwen3`, `deepseek-r1` gibi modeller → sadece **açık/kapalı**;
* `gpt-oss` → sadece **low / medium / high**, ve kapatılamaz (kutu kilitli görünür);
* bazı modellerde ek olarak **max**;
* düşünme yeteneği olmayan bir model → kontrol **gri** ve "no reasoning" yazar.

Modelin kabul etmediği bir değer isteğe hiç konmaz, böylece sunucu isteği
tümden reddetmez. Akıl yürütme metni yanıttan ayrı bir **"Reasoning"**
bloğunda akar; blok varsayılan olarak kapalıdır, tıklayarak açarsınız.

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

## 4. Kullanım limitleri (kurumsal kota)

| Limit | Varsayılan | Açıklama |
|---|---|---|
| Günlük istek | 500 | Günde gönderilebilecek LLM isteği |
| Günlük token | 1.000.000 | Girdi + çıktı toplamı |
| Oturum başına token | 100.000 | *Yeni oturum* ile sıfırlanır |
| İstek başına token | 32.000 | Çok büyük istemler gönderilmeden reddedilir |
| Günlük aktif süre | 4 saat | Modelin çalışma süresi |
| Günlük ajan koşusu | 100 | Otonom çalıştırma sayısı |
| Koşu başına araç çağrısı | 60 | Sonsuz döngü koruması |
| İstekler arası bekleme | 0 sn | Hız sınırlama |

`0` değeri **sınırsız** demektir. Sol alttaki göstergeler %80'de sarıya, dolduğunda kırmızıya döner
ve ajan çalışmayı reddeder.

### UsageLimitEditor.exe (yönetici aracı)

- Tüm kotaları düzenler, varsayılanlara döndürür.
- **Yönetici parolası** koyabilirsiniz (PBKDF2 ile saklanır); parola varsa araç açılışta sorar.
- **Developer mode**: tüm limitleri geçici olarak devre dışı bırakır.
- Son 21 günün tüketim tablosunu gösterir; bugünün sayaçlarını veya tüm geçmişi silebilir.
- Politika dosyası **AES-256-GCM** ile şifrelidir; elle kurcalanırsa Agent3 sınırsıza değil,
  **güvenli varsayılanlara** düşer.
- Ana uygulama dosya değişince politikayı **anında** yeniden okur, yeniden başlatma gerekmez.

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
limits.policy.enc    şifreli kota politikası
usage.sqlite3        kullanım telemetrisi
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
| Ajan hemen duruyor, kırmızı banner var | Kota dolmuş; `UsageLimitEditor` ile limiti yükseltin veya sayaçları sıfırlayın |
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
