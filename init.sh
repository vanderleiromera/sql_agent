#!/bin/bash
createdb -U postgres dvdrental
pg_restore -c -U postgres -d dvdrental -v "/tmp/dvdrental.tar"
psql -U postgres -d dvdrental -c "CREATE ROLE agent WITH LOGIN PASSWORD 'adminpassword';"
psql -U postgres -d dvdrental -c "GRANT CONNECT ON DATABASE dvdrental TO agent;"
psql -U postgres -d dvdrental -c "GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent;"
