# Défense du narval comme antenne acoustique à ondes fuyantes (LWA) — **chaîne v4**
## Modèle à **trois potentiels** (cylindre) : modes longitudinaux **L**, de torsion **T**, de flexion **F** — + FEM/BEM pour MeSU

Cette chaîne **adapte la chaîne v3** (plaque de Lamb, deux potentiels) au **document v4**
(décomposition de Helmholtz à **trois potentiels** sur un cylindre élastique).
Elle remplace le cœur analytique et étend le FEM/BEM, tout en restant **exécutable
sur le calculateur SACADO MeSU / MCMeSU** de Sorbonne Université.

> **Deux changements majeurs par rapport à v3**
> 1. **Physique** : le noyau « plaque de Lamb » (familles S/A, déplacements
>    `u_r,u_z` seulement) est remplacé par le **cylindre à trois potentiels**
>    `u = ∇Φ + ∇×(Ψ e_s) + ∇×∇×(χ e_s)`, qui donne les **trois familles**
>    `L(0,n)`, `T(0,n)` (torsion) et `F(m,n)` (flexion). La torsion `u_φ` était
>    **absente** de v3.
> 2. **HPC** : le run BEM v3 sur MeSU a **planté** (`Unable to allocate 415 GiB`,
>    matrice dense 166953²). La v4 utilise l'**assemblage FMM**, un **maillage
>    adaptatif par bande** et un **garde-fou mémoire** → quelques Go au lieu de
>    ~400 Go.

---

## 1. Ce que la v4 calcule (et en quoi elle diffère de v3)

| Quantité | v3 (plaque, 2 potentiels) | **v4 (cylindre, 3 potentiels)** |
|---|---|---|
| Déplacements | `u_r, u_z` | `u_r, u_φ, u_z` |
| Familles de modes | S0–S2 / A0–A2 (= `L(0,n)`) | **`L(0,n)`, `T(0,n)`, `F(m,n)`** |
| Torsion | absente | **`T(0,1)=c_T` exact, non dispersif, mode SOMBRE** |
| Rayonnement | `c_φ>c_0` | **`c_φ>c_0` ET `u_r≠0`** (torsion exclue) |
| Lecture LDV | `v_n` (normal) | `v_n, v_φ, v_s` (torsion → incidence oblique) |
| Cœur analytique | `analytic_dispersion.py` (Rayleigh–Lamb) | **`cylindrical_dispersion.py`** (Pochhammer-Chree + torsion + flexion) |
| BEM | dense (a planté) | **FMM + maillage adaptatif + garde mémoire** |

**Résultat physique central encodé :** seule la composante **normale `u_r`** se
couple au fluide parfait. La torsion (`u_r=0`) **ne rayonne pas** et est **invisible
au vibromètre en incidence normale** : c'est un *mode acoustiquement sombre*. La
condition d'antenne `c_φ>c_0` est donc **nécessaire mais non suffisante**.

---

## 2. Partie analytique (calculée et vérifiée ici)

```bash
python3 run_analytic_v4.py          # dispersion L/T/F, f-k, cg, fuite -> figures/ + data/
python3 radiation_farfield_v4.py    # diagrammes polaires 20-200 kHz (modes rayonnants)
```

### Moteur `cylindrical_dispersion.py`
- **`L(0,n)`** : équation de **Pochhammer-Chree** du cylindre plein (forme canonique
  Achenbach/Graff), évaluée en **fonctions de Bessel exponentiellement mises à
  l'échelle** (`jve`) pour rester `O(1)` et lisse sous `c_T`/`c_L`.
- **`T(0,n)`** : **exact analytique**. `T(0,1)` est **non dispersif** à `c_φ=c_T` ;
  les `T(0,n≥2)` ont pour coupures les zéros de `J₂` :
  `f_c = j_{2,n-1}·c_T/(2π R)` (≈ 41, 67, 93 kHz pour `R=28.7 mm`, `c_T=1444 m/s`).
  Marqués **`radiates=False`** (sombres).
- **`F(1,1)`** : **fondamental de flexion**, tracé par **continuation locale
  robuste** depuis une graine basse fréquence (évite l'amas de racines parasites
  du déterminant de flexion entre `c_T` et `c_L`). Les harmoniques `F(1,n≥2)` sont
  **déléguées au FEM** (voir §3) — c'est un choix assumé : leur énumération
  analytique par balayage est peu fiable, alors que le FEM les capture nativement.

### Vérifications (auto-test : `python3 cylindrical_dispersion.py`)
- `T(0,1)` = `c_T` exactement (∀f) ;
- inversion élastique `(c_L,c_T) → (E,ν)` : round-trip `E=10.30 GPa`, `ν=0.300` ;
- `L(0,1)` basse fréquence → vitesse de barre `√(E/ρ)=2328 m/s` (obtenu 2320 à 5 kHz).

### Figures (`figures/`)
- `dispersion_cph_v4.png` — `c_φ(f)` : L (bleu), F(1,1) (vert), T (gris tireté,
  « sombre »), avec `c_R`, `c_barre`, `c_T`, `c_0` et la **zone rayonnante**.
- `fk_analytic_v4.png` — diagramme f–k des trois familles + ligne d'eau.
- `dispersion_cg_v4.png` — vitesses de groupe.
- `leaky_alpha_v4.png` — fuite `α(f)` et angle de Mach `θ_M(f)` **pour les seuls
  modes rayonnants (L, F)** ; la torsion **n'y figure pas**, par construction.
- `radiation_polar_grid_v4.png`, `radiation_fmap_v4.png`,
  `radiation_polar_all_v4/*.png` — rayonnement champ lointain (L+F).

### Données (`data/`)
- `analytic_modes_v4.npz` — branches `*_f/_k/_cph` par mode + `cL,cT,cR,cbar,c0,R`.
- `leaky_modes_v4.npz` — `(f, alpha, thetaM, cph)` par branche rayonnante.
- `radiation_v4.npz` — `theta`, `|p|²(f,θ)`, points de lobe.

---

## 3. Partie numérique HPC (scripts prêts pour MeSU)

> dolfinx et bempp ne sont pas installables dans l'environnement de développement ;
> ces scripts s'exécutent sur le calculateur. Ils produisent les contreparties
> **numériques** des figures analytiques pour validation croisée.

### Dispersion FEM — `fem_dispersion_v4_hpc.py` (formulation **Fourier-mode**)
Le solveur v3 était axisymétrique `n=0` (2 composantes) → seulement `L(0,n)`.
La v4 utilise une **formulation Fourier mode-m à 3 composantes** `(u_r,u_φ,u_z)`,
`u(r,φ,z)=Re{û(r,z)e^{imφ}}`, sur le **même maillage méridien**, avec trois modes :

```bash
python fem_dispersion_v4_hpc.py --mode long      # m=0, (u_r,u_z)  -> L(0,n)  (= v3)
python fem_dispersion_v4_hpc.py --mode torsion   # m=0, u_phi      -> T(0,n)  (sombre)
python fem_dispersion_v4_hpc.py --mode flexion   # m=1, 3 comp.    -> F(1,n)
```
- chirp 20→200 kHz, intégration α-généralisé, couches absorbantes ALID ;
- **chargement fluide** appliqué à la **composante normale `u_r` uniquement**
  (couple L et F) ; **désactivé pour la torsion** (`u_r=0`, mode sombre) ;
- ligne de capteurs dense → **2-D FFT** → f–k numérique, superposé à
  `data/analytic_modes_v4.npz` (famille correspondante) ;
- sortie `fem_dispersion_v4_{long,torsion,flexion}.npz/.png`.

### Rayonnement BEM — `bem_radiation_v4_hpc.py` (**FMM, sûr en mémoire**)
**Pourquoi v3 a planté :** opérateurs assemblés **denses** + un seul maillage de
167 k nœuds (λ₀/6 à 200 kHz) → tentative d'allocation **415 GiB** (`.err` fourni).

**Corrections v4 :**
1. **Assemblage FMM** : tous les opérateurs avec `assembler="fmm"` → `O(N log N)`
   au lieu de `O(N²)` ;
2. **Maillage adaptatif par bande** : la surface est re-maillée à `λ₀(f)/6` pour
   chaque fréquence (grossier en bas de bande), nœuds plafonnés à `N_NODES_MAX` ;
3. **Garde-fou mémoire** : estimation du jeu de travail FMM avant assemblage ;
   au-delà de `MEM_BUDGET_GB`, la fréquence est grossie ou sautée proprement
   (jamais d'OOM qui tue le nœud) ;
4. surfaces temporaires créées/supprimées dans le **scratch**.

Vitesse normale imposée : celle du **mode fuyant rayonnant dominant** (L ou F),
`v_n(s)=U₀ e^{-αs} e^{iβs}`. **La torsion n'est jamais imposée** (datum de Neumann
nul → champ nul, conformément au caractère sombre).
Route 2 (couplage **two-way** FEM-BEM) : structure documentée dans le script.

### Exécution sur MeSU/MCMeSU (SLURM)
```bash
# 1) préparer l'environnement (réutilise l'env 'narwhal' v3 par défaut,
#    et y ajoute seulement le backend FMM (exafmm-t) requis par la BEM v4)
bash hpc/env_mesu.sh
# ou pour un autre env existant :    TUSK_ENV=monenv bash hpc/env_mesu.sh
# ou pour un env-prefix local       : TUSK_ENV=./tusk bash hpc/env_mesu.sh

# 2) soumettre les jobs (lisent $TUSK_ENV; même défaut 'narwhal')
sbatch hpc/job_fem_dispersion_v4.slurm      # dispersion numérique L/T/F (3 modes)
sbatch hpc/job_bem_radiation_v4.slurm       # rayonnement numérique (FMM, ~32 Go)
```

Les jobs travaillent dans `$SCRATCH` (charte MeSU : pas de calcul dans `$HOME`),
recopient les entrées, puis rapatrient les sorties dans `$SLURM_SUBMIT_DIR`.
Plus de chemin codé en dur (le v3 pointait `/home/ollivief/...`).

**Réutilisation de l'env conda existant.** `env_mesu.sh` est un **outil de
diagnostic** : il active l'env `narwhal` (ou celui défini par `$TUSK_ENV`),
sonde les paquets attendus (`dolfinx`, `bempp-cl`, `gmsh`, etc.) et **affiche
ce qui manque** sans tenter d'installer quoi que ce soit via `conda` —
parce que le solveur conda peut épuiser la RAM disponible sur la frontale
MeSU. Si vous voulez quand même tenter une installation, passez
`AUTO_INSTALL=1` (qui n'utilise alors que `pip`, beaucoup plus léger).

**Détection multi-nom de `bempp-cl`.** Selon la version, l'API s'importe
sous `bempp.api` (≤0.4.x, dont la version 0.4.2 présente dans `narwhal`) ou
sous `bempp_cl.api` (versions plus récentes). Le script de diagnostic et le
script BEM essayent les deux noms via une fonction `_import_bempp()`, ce qui
les rend portables entre les deux versions.

**Choix automatique de l'assembleur BEM.** À l'exécution, `bem_radiation_v4_hpc.py`
choisit le meilleur assembleur disponible parmi ceux exposés par bempp-cl 0.4.x :
```
fmm               (si exafmm importable ; ~3.5 ko/DOF)
  ↓ sinon
default_nonlocal  (JIT OpenCL/Numba matrix-FREE ; ~1.5 ko/DOF)
  ↓ sinon
dense             (avec garde-fou strict ; saute toute fréquence dépassant MEM_BUDGET_GB)
```
La décision est prise **paresseusement** : aucun "probe operator" coûteux
au démarrage. Si l'assembleur choisi échoue lors de la première vraie
assemblée, le script bascule sur l'option suivante et reprend. Le log
indique clairement la décision finale (`[BEM] using assembler='default_nonlocal'`
etc.).

⚠ **Note importante :** l'ancien mot-clé `hmat` (bempp 0.2.x) **n'existe pas**
dans bempp-cl 0.4.x. Le mode matrix-free `default_nonlocal` joue le rôle
équivalent ; il est encore plus économe en mémoire (la matrice complète n'est
jamais matérialisée — chaque produit matrice-vecteur du GMRES est évalué à la
volée par OpenCL/Numba), au prix d'un coût par itération plus élevé.

**Note importante : `exafmm-t` n'est pas un paquet pip et n'est pas sur
conda-forge sur MeSU.** Pour avoir spécifiquement FMM, il faut le construire
depuis les sources sur un nœud de calcul :
```bash
git clone https://github.com/exafmm/exafmm-t.git
cd exafmm-t && ./configure && make && make install && python setup.py install
```
Sinon, le BEM utilise `default_nonlocal` qui consomme ~1.5 ko/DOF (~0.1 Go
pour 60 k nœuds, contre ~50 Go en dense). Le crash à 415 GiB de v3 ne peut
donc plus revenir.

Remerciement requis dans toute publication :
> « This work was granted access to the HPC resources of the SACADO MeSU
>   platform at Sorbonne Université. »

---

## 4. Fichiers

```
cylindrical_dispersion.py     NOUVEAU moteur v4 : 3 potentiels -> L, T, F (+ fuite, inversion)
run_analytic_v4.py            pilote analytique -> dispersion L/T/F, f-k, cg, fuite
radiation_farfield_v4.py      rayonnement champ lointain (modes rayonnants L+F ; torsion exclue)
run_analytic_F11.py           pilote analytique RESTREINT au seul fondamental F(1,1)
radiation_farfield_F11.py     rayonnement RESTREINT au seul F(1,1) (lit data/F11_modes.npz)
fem_dispersion_v4_hpc.py      FEM Fourier-mode (--mode long|torsion|flexion) -> f-k numérique
bem_radiation_v4_hpc.py       BEM FMM, maillage adaptatif, garde mémoire -> rayonnement numérique
generate_axisymmetric_mesh.py maillage méridien (r,z) (inchangé v3 ; utilisé par le FEM)
postprocess_dispersion.py     2-D FFT espace-temps -> f-k (inchangé v3, vérifié)
check_mesh.py                 inspection d'un .msh surfacique (diagnostic mémoire BEM)
config.txt / config_refined.yaml   géométrie / matériaux (+ références ; inchangés)
hpc/env_mesu.sh               création de l'env conda 'tusk' (NOUVEAU)
hpc/job_fem_dispersion_v4.slurm   job FEM (3 modes, scratch)         (NOUVEAU)
hpc/job_bem_radiation_v4.slurm    job BEM (FMM, 32 Go, scratch)      (NOUVEAU)
figures/ , data/              sorties analytiques v4
```

---

## 5. Hypothèses physiques et points de vérification (v4)

- **Cylindre plein vs paroi** : le noyau v4 traite le **cylindre** (rayon de base
  `R=28.7 mm`) comme section guidante, ce qui rétablit `u_φ` (torsion, flexion).
  Un raffinement **multicouche annulaire** (canal pulpaire) se branche via une
  assemblée de matrices de transfert 6×6 (Lowe 1995) — esquissée en annexe du
  document v4 ; le FEM sur la géométrie réelle reste le juge.
- **Familles `m`** : l'analytique fournit `L(0,n)`, `T(0,n)` et le **fondamental**
  `F(1,1)`. Les harmoniques flexurales `F(1,n≥2)` et les ordres `m≥2` sont
  **délégués au FEM** (mode Fourier `m=1`, extensible à `m≥2`).
- **Torsion sombre** : `T(0,1)=c_T` exact ; **ne rayonne pas** (`u_r=0`) et est
  **invisible au LDV normal** → la mesurer exige une **incidence oblique** (EXP-6).
  Sa vitesse dépend de `c_T`, **non mesuré** : à lever par EXP-1 (la pente non
  dispersive de `T(0,1)` donne `c_T` directement).
- **`F(1,1)` basse fréquence** : limite donnée qualitativement en `√ω` (poutre) ;
  la constante exacte (Euler-Bernoulli vs Timoshenko) n'est pas figée.
- **Fuite `α`** : la dispersion libre est exacte ; `α` (méthode énergétique sur
  l'eigenfonction `u_r`) est une **estimation** à recouper par la BEM FMM.
- **Isotrope vs anisotrope** : modèle isotrope axial ; anisotropie fibreuse
  documentée (`E` hors-axe ≈ 0.45 `E_axial`) — raffinement orthotrope direct.
- **Hélice mise de côté** : corps de révolution (axisymétrique). Le couplage
  `L–T–F` induit par l'hélice (et l'éventuel moment angulaire orbital) relève de
  la 3-D complète, hors périmètre de cette chaîne.
