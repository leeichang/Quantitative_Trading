"""
系統 API 測試
"""


class TestHealth:
    """健康檢查測試"""

    def test_health_check(self, client):
        """測試健康檢查端點"""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data


class TestDataStatus:
    """資料狀態測試"""

    def test_data_status_empty_db(self, client):
        """測試空資料庫的資料狀態"""
        response = client.get("/api/v1/system/data-status")
        assert response.status_code == 200

        data = response.json()
        assert "datasets" in data
        assert len(data["datasets"]) == 6
        assert "stocks" in data
        assert "checked_at" in data

        for dataset in data["datasets"]:
            assert dataset["latest_date"] is None
            assert dataset["is_fresh"] is False
