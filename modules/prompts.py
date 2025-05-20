# Arquivo: modules/prompts.py

# Prefixo de segurança para o Agente SQL
SAFETY_PREFIX = """IMPORTANTE: Você está interagindo com um banco de dados SQL. Sua única função é gerar consultas SQL do tipo SELECT para responder às perguntas dos usuários.
NUNCA gere comandos SQL que modifiquem dados ou o schema, como INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, etc.
Sempre gere apenas comandos SELECT.
Se a pergunta do usuário implicar uma modificação ou comando não permitido, informe que você só pode executar consultas SELECT para buscar informações."""

# Exemplos few-shot para melhorar a precisão do agente SQL
FEW_SHOT_EXAMPLES = """
Aqui estão alguns exemplos de perguntas e suas respectivas consultas SQL para o banco de dados Odoo ERP:

Pergunta: Qual o total de vendas por cliente nos últimos 3 meses?
SQL:
```sql
SELECT
    rp.name AS cliente,
    SUM(so.amount_total) AS total_vendas
FROM
    sale_order so
JOIN
    res_partner rp ON so.partner_id = rp.id
WHERE
    so.date_order >= (CURRENT_DATE - INTERVAL '3 months')
    AND so.state IN ('sale', 'done')
GROUP BY
    rp.name
ORDER BY
    total_vendas DESC;
```

Pergunta: Quais são os 5 produtos mais vendidos este ano?
SQL:
```sql
SELECT
    pt.name AS produto,
    SUM(sol.product_uom_qty) AS quantidade_vendida,
    SUM(sol.price_subtotal) AS valor_total
FROM
    sale_order_line sol
JOIN
    product_product pp ON sol.product_id = pp.id
JOIN
    product_template pt ON pp.product_tmpl_id = pt.id
JOIN
    sale_order so ON sol.order_id = so.id
WHERE
    EXTRACT(YEAR FROM so.date_order) = EXTRACT(YEAR FROM CURRENT_DATE)
    AND so.state IN ('sale', 'done')
GROUP BY
    pt.name
ORDER BY
    quantidade_vendida DESC
LIMIT 5;
```

Pergunta: Qual o nível de estoque atual dos produtos que estão abaixo do ponto de reposição?
SQL:
```sql
SELECT
    pt.name AS produto,
    pt.default_code AS codigo,
    sq.quantity AS quantidade_atual,
    pt.reordering_min_qty AS ponto_reposicao,
    (pt.reordering_min_qty - sq.quantity) AS quantidade_faltante
FROM
    stock_quant sq
JOIN
    product_product pp ON sq.product_id = pp.id
JOIN
    product_template pt ON pp.product_tmpl_id = pt.id
JOIN
    stock_location sl ON sq.location_id = sl.id
WHERE
    sl.usage = 'internal'
    AND sq.quantity < pt.reordering_min_qty
    AND pt.active = true
    AND pt.type = 'product'
GROUP BY
    pt.name, pt.default_code, sq.quantity, pt.reordering_min_qty
ORDER BY
    quantidade_faltante DESC;
```

Pergunta: Qual a margem de lucro média por categoria de produto nas vendas do último trimestre?
SQL:
```sql
SELECT
    pc.name AS categoria,
    AVG((sol.price_subtotal - (sol.product_uom_qty * pt.standard_price)) / NULLIF(sol.price_subtotal, 0) * 100) AS margem_percentual
FROM
    sale_order_line sol
JOIN
    sale_order so ON sol.order_id = so.id
JOIN
    product_product pp ON sol.product_id = pp.id
JOIN
    product_template pt ON pp.product_tmpl_id = pt.id
JOIN
    product_category pc ON pt.categ_id = pc.id
WHERE
    so.date_order >= (CURRENT_DATE - INTERVAL '3 months')
    AND so.state IN ('sale', 'done')
GROUP BY
    pc.name
ORDER BY
    margem_percentual DESC;
```

Pergunta: Quais fornecedores têm o maior tempo médio de entrega para compras?
SQL:
```sql
SELECT
    rp.name AS fornecedor,
    AVG(EXTRACT(DAY FROM (sp.date_done - po.date_order))) AS tempo_medio_entrega_dias,
    COUNT(po.id) AS total_compras
FROM
    purchase_order po
JOIN
    res_partner rp ON po.partner_id = rp.id
JOIN
    stock_picking sp ON sp.origin = po.name
WHERE
    po.state = 'done'
    AND sp.state = 'done'
    AND sp.picking_type_id IN (SELECT id FROM stock_picking_type WHERE code = 'incoming')
    AND po.date_order >= (CURRENT_DATE - INTERVAL '12 months')
GROUP BY
    rp.name
HAVING
    COUNT(po.id) >= 5
ORDER BY
    tempo_medio_entrega_dias DESC
LIMIT 10;
```

pergunta: Quais produtos foram vendidos nos últimos 30 dias, mas não têm estoque em mãos?
SQL:
```sql
SELECT
    pt.name AS produto,
    SUM(sol.product_uom_qty) AS total_vendido,
    COALESCE(SUM(sq.quantity), 0) AS estoque
FROM
    sale_order_line sol
JOIN
    product_product pp ON sol.product_id = pp.id
JOIN
    product_template pt ON pp.product_tmpl_id = pt.id
LEFT JOIN
    stock_quant sq ON pp.id = sq.product_id AND sq.location_id = (SELECT id FROM stock_location WHERE name = 'Stock' LIMIT 1)
JOIN
    sale_order so ON sol.order_id = so.id
WHERE
    so.date_order >= NOW() - INTERVAL '30 days'  -- Filtrando para os últimos 30 dias
GROUP BY
    pt.id, pt.name, pt.default_code
HAVING SUM
    (sol.product_uom_qty) > 0 AND COALESCE(SUM(sq.quantity), 0) = 0;
```

Pergunta: Mostre o nivel de estoque de 20 produtos, mas vendidos em valor de 2025
SQL:
```sql
WITH mais_vendidos_valor AS (
    SELECT
        pp.id AS product_id,
        pt.name AS product_name,
        SUM(sol.price_total) AS valor_total_vendido
    FROM
        sale_order_line sol
    JOIN
        sale_order so ON sol.order_id = so.id
    JOIN
        product_product pp ON sol.product_id = pp.id
    JOIN
        product_template pt ON pp.product_tmpl_id = pt.id
    WHERE
        so.state IN ('sale', 'done')
        AND EXTRACT(YEAR FROM so.date_order) = 2025
    GROUP BY
        pp.id, pt.name
    ORDER BY
        valor_total_vendido DESC
    LIMIT 20
),
estoque AS (
    SELECT
        sq.product_id,
        SUM(sq.quantity - sq.reserved_quantity) AS estoque_disponivel
    FROM
        stock_quant sq
    JOIN
        stock_location sl ON sq.location_id = sl.id
    WHERE
        sl.usage = 'internal'
    GROUP BY
        sq.product_id
)
SELECT
    mv.product_name,
    mv.valor_total_vendido,
    COALESCE(e.estoque_disponivel, 0) AS estoque_atual
FROM
    mais_vendidos_valor mv
LEFT JOIN
    estoque e ON mv.product_id = e.product_id
ORDER BY
    mv.valor_total_vendido DESC;
```
"""

# Você pode adicionar outros prompts aqui no futuro, por exemplo:
# ANOTHER_PROMPT = """..."""
