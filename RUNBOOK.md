# Deep Cut : procédure d'un passage automatique

Compte TikTok `@deepcut.kpop` (nom affiché : Deep Cut). Vidéos verticales en anglais sur toute l'actu K-pop qui
intéresse les fans, tous groupes confondus. Ton : factuel, sourcé, sans parti pris, mais vivant.

Un passage produit une ou plusieurs vidéos (une par sujet) et les programme sur TikTok via Metricool.
Le nombre de vidéos et les créneaux sont donnés par le message qui lance le passage.

## 0. Règles qui passent avant tout

- **Sourcer chaque sujet.** Deux cas acceptés :
  - une **source primaire officielle** suffit pour un fait simple : annonce d'agence ou d'artiste (comeback, tournée,
    date de sortie), classement publié par son éditeur (Billboard, Circle, Oricon, Spotify), palmarès d'une cérémonie ;
  - sinon, **deux sources fiables indépendantes** : Yonhap, The Korea Herald, Korea JoongAng Daily, The Korea Times,
    Billboard, Reuters, AP, Music Business Worldwide, Variety, Rolling Stone. Obligatoire pour tout ce qui touche à un
    litige, un contrat, de l'argent ou une controverse.
  Soompi et allkpop servent à repérer les sujets, pas à les prouver, sauf s'ils relaient une source ci-dessus.
  Un sujet insuffisamment sourcé n'est **pas** traité. S'il n'y a aucun sujet valable, ne rien publier.
- **Neutralité** : aucun avis, aucun adjectif de jugement. Pour un litige, donner les deux versions (écran `sides`).
  Écrire « says », « according to », « the claim » plutôt que d'affirmer ce qui est contesté.
- **Sujets exclus** : rumeurs de couple, santé, vie privée, tout ce qui concerne un mineur, accusations non confirmées,
  décès, suicide ou automutilation, contenus à caractère sexuel. En cas de doute, ne pas traiter.
- **Aucune photo d'artiste, aucun logo d'agence ou de groupe, aucune parole de chanson, aucune musique existante.**
  Le moteur ne produit que du texte animé et des formes : ne rien ajouter d'autre.
- Pas de citation de plus de quelques mots. Reformuler.
- Mention « contenu généré par IA » : désactivée (`isAigc: false`), le contenu n'étant pas réaliste. Si un jour une
  vidéo contenait une voix ou une image réaliste, il faudrait l'activer.
- Ne jamais mettre de secret (clé, mot de passe) dans ce dépôt : il est public.
- Si une étape échoue (connecteur absent, envoi refusé, rendu illisible), **s'arrêter et le signaler** dans le compte
  rendu final. Ne pas publier une vidéo non vérifiée.

## 1. Préparer

```bash
git clone --depth 1 https://github.com/Chenlook/deepcut-videos && cd deepcut-videos
python3 -c "import numpy, PIL, scipy" || pip install --break-system-packages numpy pillow scipy
which ffmpeg
```

Lire `posted.md` : c'est le journal des sujets déjà publiés. Ne pas refaire un sujet déjà traité, sauf développement
majeur. Regarder aussi quels types d'écrans ont servi dans les trois dernières vidéos, pour varier.

## 2. Choisir les sujets

Le but est que la vidéo **intéresse un fan**, pas qu'elle soit sérieuse. Chercher l'actu K-pop des dernières 24 heures
avec des recherches variées, puis piocher dans des familles différentes :

- **Sorties et comebacks** : annonce, date, concept, premiers chiffres (préventes, vues du clip en 24 h, ventes du premier jour).
- **Classements et records** : Billboard, Circle, Oricon, Spotify, YouTube, premières fois, séries en cours.
- **Tournées et concerts** : annonces, stades, dates ajoutées, billets partis en quelques minutes.
- **Récompenses et émissions** : nominations, palmarès, victoires dans les émissions musicales, lineups de fin d'année.
- **Débuts et nouveaux groupes** : qui débute, sous quelle agence, avec quels chiffres.
- **Collaborations, mode et marques** : featurings, ambassadeurs, bandes originales, apparitions à l'étranger.
- **Coulisses de l'industrie** : stratégie des agences, chiffres d'affaires, tendances (versions d'albums, photocards).
- **Justice et contrats** : seulement quand c'est une vraie actu, et **au maximum une vidéo sur quatre**.

Règles de variété :
- jamais deux vidéos de suite dans la même famille, ni sur le même groupe ;
- sur une journée, viser des groupes et des agences différents (grands noms comme groupes montants, garçons et filles) ;
- relire `posted.md` : éviter les familles et les groupes des quatre dernières vidéos.

Classer les candidats par potentiel :
1. taille et ferveur du public concerné,
2. nouveauté (moins de 24 h, pas encore publié ici),
3. un angle qui donne envie de regarder : un chiffre surprenant, une première, un contraste, une question à laquelle
   la vidéo répond. Un sujet sans angle n'est pas retenu, même s'il est vrai et récent.

Prendre les meilleurs, dans la limite demandée, en ne gardant que ceux qui respectent la section 0.

## 3. Écrire le fichier du sujet

Créer `stories/AAAAMMJJ_slug.json`. Texte en anglais. Titres en capitales, courts. 5 à 9 écrans, 15 à 30 secondes.
Le premier écran est toujours `hook`, le dernier toujours `outro`. Varier les écrans d'une vidéo à l'autre.

```json
{
  "slug": "illit-top-album-sales",
  "section": "CHARTS",
  "source": "Billboard / Korea Herald",
  "caption": "Description TikTok, structurée comme indiqué plus bas.",
  "scenes": [ ... ]
}
```

`section` : un mot (CHARTS, COURTS, CONTRACTS, BUSINESS, TOURS, AWARDS, STREAMING...). `source` : noms courts des médias.
`seed` (facultatif) : un entier ; sans lui, il est tiré du slug. C'est lui qui fait varier fond, couleurs, transitions,
ordre d'apparition et musique.

### La description TikTok (`caption` à la racine du fichier)

Toujours la même structure, avec de vrais retours à la ligne (`\n` dans le JSON) :

```
<Accroche : la question ou le fait fort, 90 caractères max>

<Le résumé en une ou deux phrases, avec le chiffre clé. 200 caractères max.>

Sources: <Média 1> · <Média 2>

Follow @deepcut.kpop for one K-pop story at a time.

#kpop #<groupe> #<sujet> #kpopnews
```

Quatre à six hashtags, tous pertinents : `#kpop`, le groupe ou l'artiste, le thème (`#billboard`, `#worldtour`,
`#comeback`...), `#kpopnews`. Pas d'émoji en rafale, un seul au maximum. Pas de mots en capitales hors noms propres.

Dans le `caption` **d'un écran** (la phrase en bas d'écran), entourer d'astérisques le passage à mettre en vert : `"*first No. 1* on this chart."`
Garder ces phrases sous 40 caractères environ. `dur` (secondes) est facultatif sur chaque écran.

| type | champs | usage |
|---|---|---|
| `hook` | `top` (1-2 lignes), `token` (2-4 caractères en chrome géant : `#1`, `13B`, `24+`, `?`), `bottom`, `pill` | accroche, de préférence une question |
| `statement` | `kicker`, `lines` (2-4 lignes, préfixer `*` la ligne à mettre en vert), `caption` | une affirmation forte |
| `number` | `label`, `value` (nombre : il s'anime) , `prefix`, `suffix`, `under`, `caption` | un chiffre clé |
| `ranking` | `title`, `tag`, `rows` (`[rang, nom, sous-titre?]`, 6 max), `highlight` (index), `caption` | un classement |
| `grid` | `count` (30 max), `plus` (bool), `label` (2 lignes), `caption` | une quantité qu'on voit s'empiler |
| `timeline` | `title`, `events` (`[date, texte]`, 4 max), `caption` | une affaire en dates |
| `sides` | `title`, `a` et `b` (`who`, `text`), `caption` | deux versions d'un litige |
| `flow` | `title`, `nodes` (3-4 étapes), `caption` | un mécanisme, cause vers effet |
| `streak` | `big`, `title`, `count` (2-6), `labels`, `last`, `last_label`, `lines` (2), `caption` | une série |
| `items` | `title`, `labels` (3), `caption` | trois éléments illustrés (carte, affiche, stickers) |
| `versus` | `left` (2 lignes), `right` (2 lignes), `pill` | question ouverte au public, deux options neutres |
| `outro` | `lines`, `sub` | fin, qui boucle sur l'accroche |

Exemples complets : `stories/example_illit.json`, `stories/example_court.json`.

## 4. Rendre et vérifier

```bash
python3 engine.py stories/AAAAMMJJ_slug.json /tmp/check --stills   # une image par écran + sheet.jpg
python3 engine.py stories/AAAAMMJJ_slug.json out/AAAAMMJJ_HHMM_slug.mp4
```

Regarder `/tmp/check/sheet.jpg` **avant** le rendu complet : aucun texte coupé, superposé ou illisible, tout dans la zone
utile. Si un écran pose problème, raccourcir le texte dans le JSON et relancer. Vérifier que chaque fait à l'écran
correspond bien aux sources.

## 5. Mettre la vidéo en ligne pour Metricool

Les vidéos vont sur la branche `videos`, écrasée à chaque passage (Metricool en fait sa propre copie au moment de la
programmation, donc l'historique est inutile et le dépôt ne grossit pas).

```bash
rm -rf /tmp/vid && mkdir /tmp/vid && cp out/*.mp4 /tmp/vid/ && cd /tmp/vid
git init -q -b videos && git add . && git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit -q -m "Vidéos du passage"
git remote add origin https://github.com/Chenlook/deepcut-videos && git push -f origin videos
curl -sS -o /dev/null -w "%{http_code}\n" https://raw.githubusercontent.com/Chenlook/deepcut-videos/videos/NOM.mp4   # doit répondre 200
```

## 6. Programmer sur TikTok

Outil Metricool `createScheduledPost`, marque `blogId` **8092744**, fuseau **Europe/Paris**. Une publication par vidéo,
à l'heure du créneau demandé (meilleurs créneaux mesurés : 10:00, 12:00, 16:00, 18:00). L'heure doit être dans le futur.

```json
{"autoPublish": true, "descendants": [], "draft": false, "firstCommentText": "", "hasNotReadNotes": false,
 "media": ["https://raw.githubusercontent.com/Chenlook/deepcut-videos/videos/NOM.mp4"], "mediaAltText": [],
 "providers": [{"network": "tiktok"}],
 "publicationDate": {"dateTime": "AAAA-MM-JJTHH:MM:00", "timezone": "Europe/Paris"},
 "shortener": false, "smartLinkData": {"ids": []},
 "text": "<caption du sujet>",
 "tiktokData": {"disableComment": false, "disableDuet": false, "disableStitch": false,
   "privacyOption": "PUBLIC_TO_EVERYONE", "commercialContentThirdParty": false, "commercialContentOwnBrand": false,
   "title": "<accroche, 90 caractères max>", "autoAddMusic": false, "photoCoverIndex": 0, "isAigc": false}}
```

Vérifier dans la réponse que le statut TikTok est `PENDING` et que `media` pointe vers une copie Metricool.

## 7. Tenir le journal

Ajouter une ligne par vidéo dans `posted.md` (date, créneau, slug, accroche, liens des sources, écrans utilisés,
en précisant la famille de sujet et le groupe dans l'accroche si ce n'est pas évident),
committer avec le fichier du sujet et pousser sur `main`.

## 8. Compte rendu

Terminer par un résumé court : sujets publiés et créneaux, sujets écartés et pourquoi, tout problème rencontré.
