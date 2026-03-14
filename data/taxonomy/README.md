# Taxonomie des métiers (basée sur ESCO)

Ce dossier contient les artefacts de taxonomie normalisés construits à partir du jeu de données ESCO

dans `tabiya_dataset/`.

Générer avec :

python pipeline/build_taxonomy_from_tabiya.py --dataset-dir tabiya_dataset --out-dir data/taxonomy

Entrées optionnelles :

- data/taxonomy/internal_titles.csv (liste des titres internes, alias séparés par |)

- --onet-mapping data/taxonomy/onet_mapping.csv (fichier CSV de correspondance ESCO vers O*NET)

Outputs :

- esco_occupations.jsonl
- esco_skills.jsonl
- esco_occupation_skill.jsonl
- esco_occupations_hierarchy.jsonl
- esco_skills_hierarchy.jsonl
- esco_skill_relations.jsonl
- isco_groups.jsonl
- skill_groups.jsonl
- internal_title_mapping.jsonl (if internal titles provided)
- job_taxonomy.json (summary + counts)