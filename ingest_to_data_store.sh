#url="https://discoveryengine.googleapis.com/v1alpha/projects/dt-gen-ai-hackathon-dev/locations/global/collections/default_collection/dataStores?dataStoreId=cuad_v1_team_1"
#url="https://discoveryengine.googleapis.com/v1alpha/projects/dt-gen-ai-hackathon-dev/locations/global/collections/default_collection/dataStores/cuad_v1_team_1"
#url="https://discoveryengine.googleapis.com/v1alpha/projects/dt-gen-ai-hackathon-dev/locations/global/collections/default_collection/dataStores"
url="https://discoveryengine.googleapis.com/v1alpha/projects/dt-gen-ai-hackathon-dev/locations/global/collections/default_collection/dataStores?dataStoreId=cuad"



curl -X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
-H "X-Goog-User-Project: dt-gen-ai-hackathon-dev" \
"$url" \
-d '{
  "displayName": "data_store_test_from_notebook",
  "industryVertical": "GENERIC",
  "solutionTypes": ["SOLUTION_TYPE_SEARCH"],
  "contentConfig": "CONTENT_REQUIRED",
  "searchTier": "STANDARD",
  "searchAddOns": ["LLM"]
}'