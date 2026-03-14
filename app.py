def render_action_center(rows: List[Dict[str, Any]], cash: float, riskable: float, profile_name: str) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Günlük Aksiyon Merkezi</div><div class="section-sub">Portföyüne göre bugün dikkat etmen gerekenler ve yeni öneriler.</div>',
        unsafe_allow_html=True,
    )

    if not rows:
        st.info("Portföy boşsa önce Atlas başlangıç portföyünden veya yeni işlem ekranından pozisyon ekle.")
    else:
        for item in rows[:3]:
            clr = action_color(item["Durum"])
            st.markdown(
                f"<div class='alert-item'><div class='alert-dot' style='background:{clr};'></div><div><div style='font-weight:800;color:#F8FAFC;'>{item['Sembol']} · {item['Durum']}</div><div style='color:#CBD5E1;'>{item['Uyarı']}</div></div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        f"**Nakit önerisi:** {format_price(cash)} nakdin var. Bunun {format_price(riskable)} kadarı yeni fırsatlar için risk alanı olarak düşünülebilir. Kalan nakit SGOV / PPF tarafında bekleyebilir."
    )

    radar = build_radar(profile_name, "1G", 10)
    open_map = {row["Sembol"]: row for row in rows}

    yeni_firsatlar: List[Dict[str, Any]] = []
    artirilabilirler: List[Dict[str, Any]] = []

    if not radar.empty:
        for _, item in radar.iterrows():
            sembol = str(item["Sembol"])

            if sembol not in open_map:
                yeni_firsatlar.append(item.to_dict())
                continue

            mevcut = open_map[sembol]
            hedef_pay_text = str(mevcut.get("Hedef Pay", "-")).replace("%", "").replace(",", ".")
            mevcut_pay_text = str(mevcut.get("% Portföy", "-")).replace("%", "").replace(",", ".")

            try:
                hedef_pay = float(hedef_pay_text) / 100.0
            except Exception:
                hedef_pay = 0.0

            try:
                mevcut_pay = float(mevcut_pay_text) / 100.0
            except Exception:
                mevcut_pay = 0.0

            skor = float(item["Atlas Skoru"])

            if hedef_pay > 0 and mevcut_pay < max(hedef_pay - 0.03, 0) and skor >= 85:
                z = item.to_dict()
                z["Mevcut Pay"] = f"%{round(mevcut_pay * 100, 2)}"
                z["Hedef Pay"] = f"%{round(hedef_pay * 100, 2)}"
                artirilabilirler.append(z)

    if yeni_firsatlar:
        st.markdown("#### Yeni Atlas Fırsatları")
        for i, row in enumerate(yeni_firsatlar[:3]):
            onerilen_tutar = max(10.0, min(riskable, cash * 0.5 if cash > 0 else 0.0))
            st.markdown(
                f"<div class='radar-card'><div class='radar-title'>{row['Sembol']} · {row['Karar']}</div><div class='radar-meta'>Atlas Skoru: %{row['Atlas Skoru']} · Risk/Getiri: {row['Risk/Getiri']} · Önerilen pozisyon: {format_price(onerilen_tutar)}</div></div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button(f"{row['Sembol']} formuna aktar", key=f"prepare_{row['Sembol']}_{i}", use_container_width=True):
                    st.session_state["portfolio_prefill_symbol"] = str(row["Sembol"])
                    st.session_state["portfolio_prefill_weight"] = "%15"
                    st.session_state["portfolio_prefill_source"] = "ATLAS TRADE"
                    st.session_state["portfolio_prefill_note"] = f"Atlas fırsatı · Skor %{row['Atlas Skoru']}"
                    st.success(f"{row['Sembol']} yeni işlem formuna aktarıldı.")
            with c2:
                st.caption(f"Stop: {format_price(row['Stop'])} · Hedef: {format_price(row['Hedef'])}")

    if artirilabilirler:
        st.markdown("#### Pozisyon Artırılabilirler")
        for i, row in enumerate(artirilabilirler[:3]):
            st.markdown(
                f"<div class='radar-card'><div class='radar-title'>{row['Sembol']} · Pozisyon artırılabilir</div><div class='radar-meta'>Atlas Skoru: %{row['Atlas Skoru']} · Mevcut Pay: {row['Mevcut Pay']} · Hedef Pay: {row['Hedef Pay']}</div></div>",
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button(f"{row['Sembol']} artırma formu", key=f"boost_{row['Sembol']}_{i}", use_container_width=True):
                    st.session_state["portfolio_prefill_symbol"] = str(row["Sembol"])
                    st.session_state["portfolio_prefill_weight"] = row["Hedef Pay"]
                    st.session_state["portfolio_prefill_source"] = "ATLAS TRADE"
                    st.session_state["portfolio_prefill_note"] = f"Atlas artırma fırsatı · Skor %{row['Atlas Skoru']}"
                    st.success(f"{row['Sembol']} artırma için yeni işlem formuna aktarıldı.")
            with c2:
                st.caption(f"Stop: {format_price(row['Stop'])} · Hedef: {format_price(row['Hedef'])}")

    if not yeni_firsatlar and not artirilabilirler:
        st.info("Şu an yeni fırsat veya artırılabilir pozisyon görünmüyor.")

    st.markdown('</div>', unsafe_allow_html=True)
