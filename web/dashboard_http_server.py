#!/usr/bin/env python3
"""
智能节能系统管理界面 - 简易HTTP服务器
支持远程访问，无需Flask依赖

使用方法:
    python web/dashboard_http_server.py --port 8080

然后访问 http://服务器IP:8080
"""
import http.server
import socketserver
import argparse
import os
from pathlib import Path


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """自定义请求处理器"""
    
    def __init__(self, *args, **kwargs):
        # 设置web目录为根
        web_dir = Path(__file__).parent
        os.chdir(web_dir)
        super().__init__(*args, **kwargs)
    
    def end_headers(self):
        # 添加CORS头，允许远程访问
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()
    
    def log_message(self, format, *args):
        # 自定义日志格式
        print(f"[{self.log_date_time_string()}] {args[0]}")


def start_server(port=8080, bind='0.0.0.0'):
    """启动HTTP服务器"""
    
    with socketserver.TCPServer((bind, port), DashboardHandler) as httpd:
        print("=" * 60)
        print("智能节能系统管理界面 v6.0")
        print("=" * 60)
        print(f"服务器地址: http://{bind}:{port}")
        print(f"本地访问: http://localhost:{port}")
        print("=" * 60)
        print("按 Ctrl+C 停止服务器")
        print("=" * 60)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='智能节能系统管理界面服务器')
    parser.add_argument('--port', type=int, default=8080, help='端口号 (默认8080)')
    parser.add_argument('--bind', default='0.0.0.0', help='绑定地址 (默认0.0.0.0)')
    
    args = parser.parse_args()
    
    start_server(args.port, args.bind)
