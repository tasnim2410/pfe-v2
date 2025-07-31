"use client";

import { useState } from "react";
import { motion } from "framer-motion";

export default function HomePage() {
  const [hoveredFeature, setHoveredFeature] = useState(null);
  
  const features = [
    {
      id: 1,
      title: "Patent Analytics",
      description: "Uncover hidden patterns and emerging technologies through AI-powered patent analysis.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z" />
        </svg>
      )
    },
    {
      id: 2,
      title: "Trend Visualization",
      description: "Transform complex data into actionable intelligence with interactive dashboards.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z" />
        </svg>
      )
    },
    {
      id: 3,
      title: "Research Insights",
      description: "Access curated research reports and technology forecasts to stay ahead of disruption.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
        </svg>
      )
    },
    {
      id: 4,
      title: "Forecasting Tools",
      description: "Predict technology adoption curves and market impacts with sophisticated models.",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M16.5 12c0-1.08-.46-2.12-1.25-2.87L18 5h-7V4l-3 3 3 3v-2h5.83c-.04.33-.06.66-.06 1 0 3.86 3.14 7 7 7s7-3.14 7-7-3.14-7-7-7c-1.9 0-3.62.76-4.88 2h-2.1C9.96 2.67 8.08 2 6 2 2.69 2 0 4.69 0 8s2.69 6 6 6c1.64 0 3.14-.69 4.22-1.78l-1.5-1.5C8.28 11.34 7.41 12 6 12c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4v.5z" />
        </svg>
      )
    }
  ];

  const stats = [
    { value: "10M+", label: "Patents Analyzed" },
    { value: "97%", label: "Accuracy Rate" },
    { value: "500+", label: "Research Reports" },
    { value: "24/7", label: "Platform Access" }
  ];

  return (
    <div className="min-h-screen bg-gray-50">


      {/* Hero Section */}
      <div className="relative bg-white pt-16 pb-32 overflow-hidden">
        <div className="absolute top-0 right-0 w-1/3 h-full bg-gradient-to-b from-gray-50 to-white -z-10"></div>
        <div className="absolute bottom-0 left-0 w-2/5 h-1/2 bg-[#BDD248] opacity-5 rounded-full transform -translate-x-1/4 translate-y-1/3 -z-10"></div>
        
        <div className="container mx-auto px-6">
          <div className="flex flex-col lg:flex-row items-center">
            <div className="lg:w-1/2 mb-16 lg:mb-0">
              <motion.h1 
                className="text-4xl md:text-5xl font-bold text-gray-900 mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                Discover the Future of <span className="text-[#BDD248]">Technology</span> Innovation
              </motion.h1>
              
              <motion.p 
                className="text-lg text-gray-600 mb-8 max-w-xl"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                Our platform empowers researchers and innovators to uncover emerging technology trends through advanced patent analytics, research insights, and predictive visualizations.
              </motion.p>
              
              <motion.div 
                className="flex flex-wrap gap-4"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <button className="bg-[#BDD248] hover:bg-[#a8bb3e] text-white font-medium py-3 px-8 rounded-md transition shadow-md hover:shadow-lg">
                  Start Free Trial
                </button>
                <button className="bg-white border border-gray-300 hover:border-[#BDD248] text-gray-800 font-medium py-3 px-8 rounded-md transition">
                  <span className="flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path>
                      <rect width="20" height="14" x="2" y="5" rx="2"></rect>
                    </svg>
                    Watch Demo
                  </span>
                </button>
              </motion.div>
            </div>
            
            <div className="lg:w-1/2 flex justify-center">
              <motion.div 
                className="relative"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="w-full max-w-lg bg-gray-900 rounded-xl shadow-2xl p-6">
                  <div className="flex justify-between items-center mb-6">
                    <div className="flex space-x-2">
                      <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                    </div>
                    <div className="text-gray-400 text-sm">Technology Trend Dashboard</div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="bg-gray-800 rounded-lg p-4">
                      <div className="text-gray-400 text-sm mb-2">AI Patents</div>
                      <div className="text-white font-bold text-xl">24,831</div>
                      <div className="text-[#BDD248] text-sm flex items-center mt-1">
                        <span>↑ 12.4%</span>
                      </div>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4">
                      <div className="text-gray-400 text-sm mb-2">Blockchain</div>
                      <div className="text-white font-bold text-xl">8,427</div>
                      <div className="text-[#BDD248] text-sm flex items-center mt-1">
                        <span>↑ 8.2%</span>
                      </div>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4">
                      <div className="text-gray-400 text-sm mb-2">Quantum</div>
                      <div className="text-white font-bold text-xl">3,219</div>
                      <div className="text-[#BDD248] text-sm flex items-center mt-1">
                        <span>↑ 21.7%</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-800 rounded-lg p-4 mb-4">
                    <div className="flex justify-between items-center mb-4">
                      <div className="text-white font-medium">Emerging Tech Trends</div>
                      <div className="text-gray-400 text-sm">Last 12 months</div>
                    </div>
                    <div className="h-48 flex items-end space-x-1">
                      {[40, 65, 80, 55, 75, 90, 70, 85, 65, 50, 75, 95].map((height, index) => (
                        <div key={index} className="flex-1 flex flex-col items-center">
                          <div 
                            className={`w-full rounded-t ${
                              index === 11 ? 'bg-[#BDD248]' : 'bg-gray-700'
                            }`} 
                            style={{ height: `${height}%` }}
                          ></div>
                          <div className="text-gray-500 text-xs mt-1">{['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'][index]}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="flex justify-between text-gray-500 text-sm">
                    <div>Updated just now</div>
                    <div>Source: Global Patent Database</div>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="py-12 bg-gray-900 text-white">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div 
                key={index}
                className="text-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <div className="text-3xl font-bold text-[#BDD248] mb-2">{stat.value}</div>
                <div className="text-gray-300">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-20 bg-white">
        <div className="container mx-auto px-6">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Technology Intelligence Platform</h2>
            <p className="text-gray-600">Our platform integrates advanced analytics with intuitive visualization to help you stay ahead of technology trends.</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
            {features.map((feature) => (
              <motion.div 
                key={feature.id}
                className="relative"
                onMouseEnter={() => setHoveredFeature(feature.id)}
                onMouseLeave={() => setHoveredFeature(null)}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className={`p-1 rounded-xl ${hoveredFeature === feature.id ? 'bg-gradient-to-r from-[#BDD248] to-[#a8bb3e]' : 'bg-gray-100'}`}>
                  <div className="bg-white p-8 rounded-lg h-full">
                    <div className={`w-16 h-16 rounded-lg mb-6 flex items-center justify-center ${
                      hoveredFeature === feature.id ? 'bg-[#BDD248] text-white' : 'bg-gray-100 text-gray-700'
                    }`}>
                      {feature.icon}
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                    <p className="text-gray-600">{feature.description}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Process Section */}
      <div className="py-20 bg-gray-50">
        <div className="container mx-auto px-6">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-gray-600">Transform raw data into actionable insights in three simple steps</p>
          </div>
          
          <div className="flex flex-col md:flex-row justify-between items-start max-w-5xl mx-auto">
            <div className="flex flex-col items-center text-center mb-16 md:mb-0 md:w-1/3 relative">
              <div className="w-20 h-20 rounded-full bg-white border-4 border-[#BDD248] flex items-center justify-center text-gray-900 text-2xl font-bold mb-6 z-10">1</div>
              <div className="absolute top-10 left-1/2 transform -translate-x-1/2 w-0.5 h-40 bg-gray-200"></div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Connect Data</h3>
              <p className="text-gray-600">Integrate patent databases, research papers, and market reports</p>
            </div>
            
            <div className="flex flex-col items-center text-center mb-16 md:mb-0 md:w-1/3 relative">
              <div className="w-20 h-20 rounded-full bg-white border-4 border-[#BDD248] flex items-center justify-center text-gray-900 text-2xl font-bold mb-6 z-10">2</div>
              <div className="absolute top-10 left-1/2 transform -translate-x-1/2 w-0.5 h-40 bg-gray-200"></div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Analyze Trends</h3>
              <p className="text-gray-600">Use AI-powered tools to uncover patterns and visualize technology trends</p>
            </div>
            
            <div className="flex flex-col items-center text-center md:w-1/3">
              <div className="w-20 h-20 rounded-full bg-white border-4 border-[#BDD248] flex items-center justify-center text-gray-900 text-2xl font-bold mb-6 z-10">3</div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Generate Insights</h3>
              <p className="text-gray-600">Create comprehensive reports and forecasts to guide innovation strategy</p>
            </div>
          </div>
        </div>
      </div>

      {/* Testimonials
      <div className="py-20 bg-white">
        <div className="container mx-auto px-6">
          <div className="text-center max-w-2xl mx-auto mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Trusted by Innovators</h2>
            <p className="text-gray-600">Join thousands of researchers, analysts, and technology leaders</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            <div className="bg-gray-50 rounded-xl p-6">
              <div className="text-gray-600 mb-4">
                "This platform transformed how we track emerging technologies. The patent analytics feature alone saved us months of research time."
              </div>
              <div className="flex items-center">
                <div className="w-10 h-10 rounded-full bg-gray-300 mr-3"></div>
                <div>
                  <div className="font-bold text-gray-900">Dr. Sarah Chen</div>
                  <div className="text-gray-500 text-sm">Research Director, Tech Innovations</div>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-xl p-6">
              <div className="text-gray-600 mb-4">
                "The forecasting tools helped us identify quantum computing trends two years before our competitors. Game changing insights."
              </div>
              <div className="flex items-center">
                <div className="w-10 h-10 rounded-full bg-gray-300 mr-3"></div>
                <div>
                  <div className="font-bold text-gray-900">Michael Rodriguez</div>
                  <div className="text-gray-500 text-sm">CTO, FutureTech Labs</div>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 rounded-xl p-6">
              <div className="text-gray-600 mb-4">
                "Our investment decisions are now data-driven thanks to the comprehensive technology intelligence this platform provides."
              </div>
              <div className="flex items-center">
                <div className="w-10 h-10 rounded-full bg-gray-300 mr-3"></div>
                <div>
                  <div className="font-bold text-gray-900">Elena Petrova</div>
                  <div className="text-gray-500 text-sm">VC Partner, Innovation Capital</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div> */}

      {/* CTA Section */}
      <div className="py-16 bg-gradient-to-r from-gray-900 to-black">
        <div className="container mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold text-white mb-6">Ready to Uncover Technology Trends?</h2>
          <p className="text-gray-300 max-w-2xl mx-auto mb-8 text-lg">Join thousands of researchers and innovators who use our platform to stay ahead of the curve.</p>
          
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <button className="bg-[#BDD248] hover:bg-[#a8bb3e] text-gray-900 font-medium py-3 px-8 rounded-md transition shadow-md">
              Start Free Trial
            </button>
            <button className="bg-transparent border border-gray-500 hover:border-[#BDD248] text-white font-medium py-3 px-8 rounded-md transition">
              Request Demo
            </button>
          </div>
          
          <div className="mt-8 text-gray-400 text-sm">
            No credit card required • 14-day free trial • Cancel anytime
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-900 text-gray-400 py-12">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-[#BDD248] rounded-lg flex items-center justify-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                  </svg>
                </div>
                <span className="text-xl font-bold text-white">TechTrend</span>
              </div>
              <p className="text-sm mb-4">
                Empowering innovation through data-driven technology intelligence.
              </p>
              <div className="flex space-x-4">
                <a href="#" className="hover:text-[#BDD248] transition">
                  <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24"><path d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z"/></svg>
                </a>
                <a href="#" className="hover:text-[#BDD248] transition">
                  <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                </a>
              </div>
            </div>
            
            <div>
              <h4 className="text-white font-medium mb-4">Platform</h4>
              <ul className="space-y-2">
                <li><a href="#" className="hover:text-[#BDD248] transition">Features</a></li>
                <li><a href="#" className="hover:text-[#BDD248] transition">Solutions</a></li>
                <li><a href="#" className="hover:text-[#BDD248] transition">Pricing</a></li>
                <li><a href="#" className="hover:text-[#BDD248] transition">Demo</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-white font-medium mb-4">Resources</h4>
              <ul className="space-y-2">
                <li><a href="#" className="hover:text-[#BDD248] transition">Blog</a></li>
                <li><a href="#" className="hover:text-[#BDD248] transition">Documentation</a></li>
                <li><a href="#" className="hover:text-[#BDD248] transition">Guides</a></li>
                <li><a href="#" className="hover:text-[#BDD248] transition">API</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-white font-medium mb-4">Company</h4>
              <ul className="space-y-2">
                <li><a href="#" className="hover:text-[#BDD248] transition">About Us</a></li>
                <li><a href="#" className="hover:text-[#BDD248] transition">Careers</a></li>
                <li><a href="#" className="hover:text-[#BDD248] transition">Contact</a></li>
                <li><a href="#" className="hover:text-[#BDD248] transition">Partners</a></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-800 mt-12 pt-8 text-center text-sm">
            <p>© 2023 TechTrend Analytics. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}