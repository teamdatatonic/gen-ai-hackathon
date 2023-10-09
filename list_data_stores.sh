url="https://discoveryengine.googleapis.com/v1alpha/projects/dt-gen-ai-hackathon-dev/locations/global/collections/default_collection/dataStores"

curl -X GET \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
-H "X-Goog-User-Project: dt-gen-ai-hackathon-dev" \
"$url"