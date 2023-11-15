#url="https://discoveryengine.googleapis.com/v1alpha/projects/dt-gen-ai-hackathon-dev/locations/global/collections/default_collection/engines?engineId=alphabet-investor-pdfs-tea_1696409737598"
#url="https://discoveryengine.googleapis.com/v1alpha/projects/dt-gen-ai-hackathon-dev/locations/global/collections/default_collection/engines"

curl -X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
-H "X-Goog-User-Project: dt-gen-ai-hackathon-dev" \
"$url" \
-d '{
  "displayName": "app_team_2",
  "dataStoreIds": "alphabet-investor-pdfs-tea_1696409737598",
  "solutionType": "SOLUTION_TYPE_SEARCH"
}'