-- Run this once in the Supabase SQL Editor, then click Run.
-- The publishable key cannot create tables, so this step has to happen in the dashboard.

create table if not exists public.analyses (
    id bigint generated always as identity primary key,
    created_at timestamptz not null default now(),
    model_variant text not null,
    volume_ml double precision,
    concentration double precision not null,
    predicted_absorbance double precision not null,
    predicted_r integer not null,
    predicted_g integer not null,
    predicted_b integer not null,
    predicted_hex text not null,
    analyzed_r integer not null,
    analyzed_g integer not null,
    analyzed_b integer not null,
    analyzed_hex text not null,
    rgb_distance double precision,
    image_path text
);

grant select, insert on table public.analyses to anon, authenticated;
grant usage, select on all sequences in schema public to anon, authenticated;

alter table public.analyses enable row level security;

drop policy if exists "lab read analyses" on public.analyses;
create policy "lab read analyses"
    on public.analyses
    for select
    to anon, authenticated
    using (true);

drop policy if exists "lab insert analyses" on public.analyses;
create policy "lab insert analyses"
    on public.analyses
    for insert
    to anon, authenticated
    with check (true);

notify pgrst, 'reload schema';
